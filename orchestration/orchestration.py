#!/usr/bin/env python3
"""
orchestrator.py

Voice assistant orchestration service.

Listens for Wyoming Transcript events from the STT server, extracts intent
via a small Ollama model, dispatches to the appropriate skill handler, then
hands the response text off to the TTS service.

For fixed/predictable responses (unknown intent, handler errors, etc.) the
audio is pre-synthesised at startup and sent directly to the satellite as raw
PCM — no TTS round-trip needed. See audio_io.py.

On a positive intent match, an acknowledgement sound is fired to the satellite
immediately while the LLM and handler run in the background, reducing apparent
latency.

Wyoming event flow (inbound from STT server):
    Transcript  — carries transcribed text; satellite peer IP in the data field

Wyoming event flow (outbound to satellite):
    AudioStart  — declares sample rate / width / channels
    AudioChunk  — raw PCM bytes
    AudioStop   — signals end of audio

Intent JSON schema (produced by small Ollama model):
    {
        "intent": "<intent_name>",
        "slots": { ... },          # intent-specific key/value pairs
        "target_satellites": [ ]   # room names the response should play in;
                                   # empty/omitted means "wherever this was heard"
    }

target_satellites is extracted once, at the top level, for every intent —
it is routing metadata, not part of any skill's own slots. See
_resolve_target_satellites() for how room names become satellite IPs and
skills/base.py's Skill docstring for how it reaches skill handlers.

Supported intents are not hard-coded here — they're whatever's registered in
skills/registry.py. Each entry in that list documents its own intent name and
slots; see skills/README.md to add a new one.

Dependencies:
    pip install wyoming
    Ollama must be running and reachable at OLLAMA_BASE_URL, with both
    INTENT_MODEL and intent_router.py's EMBED_MODEL pulled.
    Pre-synthesised .wav files must exist at the configured paths before startup.
"""

import asyncio
import json
import logging
import sys
import urllib.request
import urllib.error
from datetime import datetime, timezone
from typing import Callable

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event, async_read_event, async_write_event

import audio_io
import conversation_state
import database
import intent_router
from conversation_state import ClarificationNeeded
from skills import music as music_skill
from skills.base import SlotSpec, parse_value_string
from skills.registry import REGISTERED_SKILLS, TRIGGER_HANDLERS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: address and port this server listens on (STT server connects here)
HOST = "127.0.0.1"
PORT = 10301

# CONFIGURE: address and port satellites connect to for the persistent
# satellite link (presence reports today — see satellite_link.py on the
# satellite side, and handle_satellite_link_connection() below). A
# separate listener from HOST/PORT above on purpose: that one speaks
# Wyoming events and is only ever contacted by the STT server (hence
# defaulting to loopback); this one is satellite-initiated, from real
# devices elsewhere on the network, so it needs to bind every interface.
SATELLITE_LINK_HOST = "0.0.0.0"
SATELLITE_LINK_PORT = 10303

# CONFIGURE: satellite name → IP address mapping
# Assign static DHCP leases to your Pis so these don't drift.
SATELLITES: dict[str, str] = {
    "living_room": "192.168.1.10",
    "kitchen":     "192.168.1.11",
    "bedroom":     "192.168.1.12",
}

# CONFIGURE: Ollama base URL
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# CONFIGURE: small model for intent extraction
INTENT_MODEL = "qwen3:1.7b"

# CONFIGURE: address and port of the TTS service
TTS_HOST = "127.0.0.1"
TTS_PORT = 10302

# CONFIGURE: how often the trigger poller wakes to check for due triggers
# (seconds). 5 seconds gives acceptable precision without hammering the DB.
# Shared by every skill that schedules future events (timers today; alarms,
# scheduled lights, etc. later) — there's one poller, not one per skill.
TRIGGER_POLL_INTERVAL = 5

# CONFIGURE: how often presence_pruner() sweeps stale presence data.
# Kept roughly in line with database.STALE_SECONDS so a dead/out-of-range
# tag's resolved location doesn't linger far longer than a sighting is
# considered fresh for.
PRESENCE_PRUNE_INTERVAL_S = 30

# CONFIGURE: how often media_follow_poller() checks whether any
# follow-enabled media session needs to move rooms. Combined with
# database.STALE_SECONDS/MARGIN_DB's own dwell time, total lag before
# music follows a person into a new room is roughly the sum of both —
# tune independently: presence hysteresis affects *correctness* of room
# detection, this affects *responsiveness* of the follow action.
MEDIA_FOLLOW_POLL_INTERVAL_S = 10

# Clarification timeout and max turns are configured in conversation_state.py.
# Pre-synthesised response config (RESPONSES_DIR, RESPONSE_FILES) lives in
# audio_io.py.

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TTS handoff
# ---------------------------------------------------------------------------

async def synthesize_speech(text: str) -> tuple[bytes, int, int, int]:
    """
    Send `text` to the TTS service and collect the synthesised PCM audio.

    Returns (pcm_bytes, sample_rate, sample_width, channels). Raises
    ConnectionError or OSError on failure — callers decide how to handle
    that (e.g. fall back to a canned error response).
    """
    reader, writer = await asyncio.open_connection(TTS_HOST, TTS_PORT)
    try:
        await async_write_event(
            Event(type="synthesize", data={"text": text}),
            writer,
        )

        pcm = bytearray()
        rate = width = channels = None

        while True:
            event = await async_read_event(reader)
            if event is None:
                raise ConnectionError("TTS service closed connection mid-stream")

            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
                rate, width, channels = start.rate, start.width, start.channels

            elif AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                pcm.extend(chunk.audio)

            elif AudioStop.is_type(event.type):
                break

        if rate is None:
            raise ConnectionError("TTS service never sent AudioStart")

        return bytes(pcm), rate, width, channels
    finally:
        writer.close()
        await writer.wait_closed()


async def send_to_tts(text: str, satellite_ip: str, listen_after: bool = False) -> None:
    """
    Synthesise `text` via the TTS service, then deliver the result to the
    satellite once synthesis is complete.

    Audio always comes back here rather than being sent directly from the TTS
    service — this keeps orchestration in control of playback ordering against
    earcons and other audio already queued for the same satellite.

    If listen_after is True, the satellite opens the mic immediately after
    playback without requiring a new wake word activation.
    """
    pcm, rate, width, channels = await synthesize_speech(text)
    await audio_io.send_audio_to_satellite(pcm, rate, width, channels, satellite_ip, listen_after=listen_after)


# ---------------------------------------------------------------------------
# Intent-extraction prompt — assembled from the registered skills
# ---------------------------------------------------------------------------
# The header/footer are the only parts that are genuinely shared across every
# possible skill. Everything intent-specific — including formatting rules
# like "always return an array" — lives in that skill's own prompt_block, so
# a skill is fully self-contained and adding one never means editing this
# file. See skills/base.py's Skill docstring for the expected block style.

_INTENT_PROMPT_HEADER = """\
You are an intent extraction engine for a home voice assistant.

Given a voice command transcript, return ONLY a JSON object — no prose, no
markdown fences — in this exact shape:

{
  "intent": "<intent_name>",
  "slots": {},
  "target_satellites": []
}

target_satellites is a JSON array of room names naming where the response
should be played, e.g. "set a timer in the kitchen" -> ["kitchen"]. Only
include a room if the user actually named one — if no room is mentioned,
return an empty array; the response automatically goes back to whichever
room heard the command. A user may name more than one room.

Supported intents and their required slots:
"""

_INTENT_PROMPT_FOOTER = """
Rules:
- Choose exactly one intent.
- Only include slots defined for that intent.
- Only include a target_satellites entry if it exactly matches one of the
  valid room names above; omit anything you don't recognise rather than
  guessing.
- Do not add commentary or explanation.
"""


def _build_intent_system_prompt(skills: list) -> str:
    blocks = "\n".join(skill.prompt_block for skill in skills)
    satellite_line = f"Valid room names: {', '.join(sorted(SATELLITES))}.\n"
    return _INTENT_PROMPT_HEADER + "\n" + satellite_line + "\n" + blocks + _INTENT_PROMPT_FOOTER


# Built once from every registered skill — the fallback system prompt used
# whenever intent_router.shortlist() opts out of filtering (not ready,
# mid-run embedding failure, or low routing confidence for this
# transcript). See extract_intent() below for the per-request path, which
# is what actually runs on a normal, confident request.
_FULL_INTENT_SYSTEM_PROMPT = _build_intent_system_prompt(REGISTERED_SKILLS)


def _call_ollama(transcript: str, system_prompt: str) -> dict:
    """
    POST to the Ollama /api/chat endpoint and return the parsed JSON intent.

    `system_prompt` is built fresh per request from whichever skills
    intent_router.shortlist() selected (or _FULL_INTENT_SYSTEM_PROMPT, if
    it opted out of filtering for this request) — see extract_intent().

    Raises urllib.error.URLError on network failure, json.JSONDecodeError on
    bad model output, KeyError if the response shape is unexpected. All
    propagate to the caller.
    """
    payload = json.dumps({
        "model": INTENT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": transcript},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        # Qwen3 is a hybrid reasoning model and thinks by default, which adds
        # several seconds of hidden <think> tokens before the JSON output —
        # never needed for slot-style intent extraction.
       "think": False,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())

    intent_json = json.loads(body["message"]["content"])
    return intent_json


def extract_intent(transcript: str) -> dict:
    """
    Ask the small LLM to extract intent from `transcript`.

    Only the skills intent_router.shortlist() selects for this transcript
    are included in the system prompt — see that module's docstring for
    why this is a shortlist, not a hard filter, and why it fails open
    (falls back to every registered skill) whenever it isn't confident.

    Returns a dict with at minimum an "intent" key. On any failure, logs the
    error and returns {"intent": "unknown", "slots": {}}.
    """

    shortlisted = intent_router.shortlist(transcript)
    if shortlisted is None:
        system_prompt = _FULL_INTENT_SYSTEM_PROMPT
    else:
        system_prompt = _build_intent_system_prompt(shortlisted)

    try:
        result = _call_ollama(transcript, system_prompt)
        log.debug("Intent extracted: %s", result)
        return result
    except (urllib.error.URLError, TimeoutError) as exc:
        log.error("Ollama unreachable: %s", exc)
    except json.JSONDecodeError as exc:
        log.error("Ollama returned non-JSON: %s", exc)
    except KeyError as exc:
        log.error("Unexpected Ollama response shape, missing key: %s", exc)

    return {"intent": "unknown", "slots": {}, "target_satellites": []}


# ---------------------------------------------------------------------------
# Slot filling — used during clarification turns
# ---------------------------------------------------------------------------

# The slot-fill prompt is deliberately narrow: it only tries to extract one
# named slot from a short reply. No intent classification happens here — the
# intent is already known from the pending context.
#
# Slots are registered per (intent, slot_name) pair rather than by slot_name
# alone — see skills/base.py's Skill docstring for why. SLOT_FILL_REGISTRY
# here is just the union of every registered skill's own slot_specs,
# namespaced by that skill's intent; a skill never has to know or care what
# slot names any other skill uses.

SLOT_FILL_REGISTRY: dict[tuple[str, str], SlotSpec] = {
    (skill.intent, slot_name): spec
    for skill in REGISTERED_SKILLS
    for slot_name, spec in skill.slot_specs.items()
}

# Fallback for an (intent, slot) pair that hasn't been registered — e.g. a
# skill author raised ClarificationNeeded for a slot they forgot to add to
# slot_specs. Generic single-string handling, but logs a warning so the gap
# is visible instead of silently defaulting forever.
_DEFAULT_SLOT_SPEC = SlotSpec(
    description="Extract the requested value. Return a JSON object with key: value.",
    parse=parse_value_string,
)


def _get_slot_spec(intent: str, slot: str) -> SlotSpec:
    spec = SLOT_FILL_REGISTRY.get((intent, slot))
    if spec is None:
        log.warning(
            "No SlotSpec registered for (intent=%r, slot=%r) — falling back to "
            "generic single-string handling. Add an entry to this skill's "
            "slot_specs if it needs different parsing.",
            intent, slot,
        )
        return _DEFAULT_SLOT_SPEC
    return spec


def _slot_fill_system_prompt(intent: str, missing_slot: str) -> str:
    spec = _get_slot_spec(intent, missing_slot)
    return (
        "You are a slot extraction engine for a home voice assistant.\n\n"
        "The user is responding to a clarifying question. "
        "Extract only the requested value from their reply.\n\n"
        f"What to extract: {spec.description}\n\n"
        "Return ONLY a JSON object — no prose, no markdown fences."
    )


def _call_ollama_slot_fill(reply: str, intent: str, missing_slot: str) -> dict:
    """
    Ask the LLM to extract one named slot from a short clarifying reply.

    Returns the raw parsed JSON dict. Raises on network failure, bad JSON,
    or unexpected response shape — callers handle these.
    """
    payload = json.dumps({
        "model": INTENT_MODEL,
        "messages": [
            {"role": "system", "content": _slot_fill_system_prompt(intent, missing_slot)},
            {"role": "user",   "content": reply},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "think": False,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())

    return json.loads(body["message"]["content"])


def _merge_slot_from_fill(
    intent: str,
    missing_slot: str,
    fill_result: dict,
    existing_slots: dict,
) -> dict:
    """
    Deterministically merge a slot-fill result into the existing slots dict.

    Looks up the (intent, missing_slot) pair in SLOT_FILL_REGISTRY and uses
    its parse function to turn fill_result into a value.

    Returns the updated slots dict. Raises ValueError if the fill_result is
    unusable (e.g. all zeros for a duration, or blank/empty for a name or
    item list) — propagated from the slot's parse function.
    """
    spec = _get_slot_spec(intent, missing_slot)
    slots = dict(existing_slots)  # don't mutate the original
    slots[missing_slot] = spec.parse(fill_result)
    return slots


# ---------------------------------------------------------------------------
# Dispatch table — built from the registered skills
# ---------------------------------------------------------------------------

HANDLERS: dict[str, Callable] = {skill.intent: skill.handler for skill in REGISTERED_SKILLS}


async def dispatch(
    intent: dict, satellite_ip: str, target_satellites: list[str], user_id: str,
) -> None:
    """
    Fire the acknowledgement earcon, then route to the correct skill handler.

    The ack is sent as a background task so it plays while the handler runs.
    The handler's return value (if not None) is broadcast via TTS to every
    IP in target_satellites.

    satellite_ip is always the satellite that heard the command (the
    origin) — the ack, clarification questions, and error responses always
    go there regardless of target_satellites, so the person who's talking
    always gets feedback even if their actual request was routed elsewhere.

    target_satellites must already be resolved to satellite IPs (see
    _resolve_target_satellites) before calling this — dispatch() itself
    does no name lookup, so a clarification follow-up can pass the same
    list it captured on the first turn without re-resolving anything.

    user_id is the speaker identified by the STT server's speaker-ID step
    ("unknown" if unidentified or below its confidence threshold). It is
    passed straight through as the handler's fourth positional parameter —
    every registered handler must accept it (skills/registry.py checks this
    at import time) — rather than being folded into slots or looked up via
    signature inspection on every call. Most skills ignore the value
    entirely, the same way most ignore target_satellites; a skill only
    needs to read it if behaviour or persisted data should vary by who's
    asking (e.g. "play my playlist").

    If the handler raises ClarificationNeeded, the question is spoken to the
    user and the context (including target_satellites and user_id) is stored
    so the next utterance can fill the missing slot without repeating intent
    extraction or losing who originally asked.
    """
    intent_name = intent.get("intent", "unknown")
    slots = intent.get("slots", {})
    handler = HANDLERS.get(intent_name, HANDLERS["unknown"])

    satellite_name = satellite_name_from_ip(satellite_ip) or satellite_ip
    log.info(
        "Dispatching intent=%r slots=%s from %s (user=%r) -> targets=%s",
        intent_name, slots, satellite_name, user_id, target_satellites,
    )

    # Fire acknowledgement immediately for known intents — runs concurrently
    # with the handler so the satellite has audio feedback while we wait on
    # the TTS. Unknown intent gets no ack; it will play its own canned
    # response instead. Ack always plays at the origin — it means "I heard
    # you", which is true regardless of where the eventual response goes.
    if intent_name != "unknown":
        ack_task = asyncio.create_task(
            audio_io.send_canned("acknowledged", satellite_ip),
            name=f"ack-{satellite_ip}",
        )
    else:
        ack_task = None

    try:
        response_text = await handler(slots, satellite_ip, target_satellites, user_id)
    except ClarificationNeeded as clarification:
        # Handler needs more information. Store context, speak the question.
        log.info(
            "Clarification needed for intent=%r missing_slot=%r: %r",
            clarification.intent, clarification.missing_slot, clarification.question,
        )
        ctx = conversation_state.ClarificationContext(
            intent=clarification.intent,
            slots=clarification.slots,
            missing_slot=clarification.missing_slot,
            question=clarification.question,
            target_satellites=target_satellites,
            user_id=user_id,
        )
        conversation_state.set(satellite_ip, ctx)

        if ack_task is not None:
            await ack_task
        try:
            # listen_after=True — satellite opens mic immediately after the
            # question finishes playing, no wake word needed for the reply.
            # Always asked at the origin — the clarification is a
            # conversation with whoever spoke, not with target_satellites.
            await send_to_tts(clarification.question, satellite_ip, listen_after=True)
        except (ConnectionError, OSError):
            log.exception("TTS unreachable when asking clarification question")
            await audio_io.send_canned("error", satellite_ip)
        return
    except Exception:
        log.exception("Handler for %r raised an exception", intent_name)
        if ack_task is not None:
            await ack_task
        await audio_io.send_canned("error", satellite_ip)
        return

    # Ensure the ack has finished before TTS audio starts playing.
    if ack_task is not None:
        await ack_task

    if response_text is not None:
        await _broadcast_response(response_text, target_satellites, origin_ip=satellite_ip)


async def _broadcast_response(text: str, target_satellites: list[str], origin_ip: str) -> None:
    """
    Send `text` to every satellite in target_satellites.

    A failure delivering to one target is logged and does not stop delivery
    to the others. If any delivery fails, a single canned error is sent to
    origin_ip afterwards — the speaker should always find out something
    went wrong even if the request was routed elsewhere and they never
    heard the failure themselves.
    """
    any_failed = False
    for target_ip in target_satellites:
        try:
            await send_to_tts(text, target_ip)
        except (ConnectionError, OSError):
            log.exception("TTS service unreachable or failed delivering to %s", target_ip)
            any_failed = True

    if any_failed:
        try:
            await audio_io.send_canned("error", origin_ip)
        except (ConnectionError, OSError):
            log.exception(
                "Could not reach origin satellite %s to report a delivery failure",
                origin_ip,
            )

# ---------------------------------------------------------------------------
# Trigger polling loop
# ---------------------------------------------------------------------------
# Generic across every skill that schedules a future event — this function
# has no idea what a "timer" is. It knows how to find due rows, how to ask
# the right skill what to say about one, and how to get that text to a
# satellite. What the event means is entirely up to the skill that scheduled
# it; see skills/base.py's TriggerHandler and skills/timer.py for the one
# built-in example.

async def trigger_poller() -> None:
    """
    Background task. Wakes every TRIGGER_POLL_INTERVAL seconds, checks for
    due triggers in the database, asks the owning skill's TriggerHandler
    for announcement text, and broadcasts it to every IP in the trigger's
    target_satellites (falling back to origin_satellite_ip for error
    reporting, same as the immediate-response path in dispatch()).

    Runs for the lifetime of the process. Errors on any individual trigger
    (missing handler, handler exception, TTS failure) are logged and
    skipped — they never stop the poller or affect other due triggers.
    """
    log.info("Trigger poller started (interval=%ds)", TRIGGER_POLL_INTERVAL)

    while True:
        await asyncio.sleep(TRIGGER_POLL_INTERVAL)

        now = datetime.now(timezone.utc)
        due = database.get_due_triggers(now)

        for trigger in due:
            log.info(
                "Trigger fired: id=%d skill=%r trigger_key=%r origin=%s user=%r targets=%s",
                trigger.id, trigger.skill, trigger.trigger_key,
                trigger.origin_satellite_ip, trigger.user_id, trigger.target_satellites,
            )

            # Mark fired immediately — if the handler or TTS fails below, we
            # still don't want to re-announce it on the next poll cycle.
            database.mark_trigger_fired(trigger.id)

            handler = TRIGGER_HANDLERS.get(trigger.skill)
            if handler is None:
                # A trigger row exists for a skill name with no registered
                # TriggerHandler — most likely a skill was removed from
                # SKILL_MODULES (or renamed its skill_name) while old rows
                # for it were still pending. Nothing sensible to announce.
                log.error(
                    "No TriggerHandler registered for skill=%r (trigger id=%d) — "
                    "dropping. Was this skill removed from skills/registry.py's "
                    "SKILL_MODULES, or its TRIGGER.skill_name renamed?",
                    trigger.skill, trigger.id,
                )
                continue

            try:
                announcement = await handler.on_trigger(trigger.payload, trigger.user_id)
            except Exception:
                log.exception(
                    "TriggerHandler for skill=%r raised on trigger id=%d — announcement lost",
                    trigger.skill, trigger.id,
                )
                continue

            if announcement is None:
                # Skill handled its own output (or has nothing to say).
                continue

            await _broadcast_response(
                announcement, trigger.target_satellites, origin_ip=trigger.origin_satellite_ip,
            )


# ---------------------------------------------------------------------------
# Presence pruning loop
# ---------------------------------------------------------------------------
# Silent housekeeping, deliberately its own sibling task rather than folded
# into trigger_poller() or media_follow_poller() — a bug in presence
# pruning shouldn't be able to delay a timer announcement or a room move.

async def presence_pruner() -> None:
    """
    Background task. Wakes every PRESENCE_PRUNE_INTERVAL_S seconds and
    calls database.prune_stale_presence() — clearing any resolved location
    no longer backed by a fresh sighting, and garbage-collecting old raw
    sightings. See that function's docstring for why staleness is only
    ever evaluated there, never on the request path.

    Runs for the lifetime of the process. A failed sweep is logged and
    retried next interval rather than raising — the same fail-open
    philosophy as trigger_poller().
    """
    log.info("Presence pruner started (interval=%ds)", PRESENCE_PRUNE_INTERVAL_S)

    while True:
        await asyncio.sleep(PRESENCE_PRUNE_INTERVAL_S)
        try:
            database.prune_stale_presence()
        except Exception:
            log.exception("presence_pruner: sweep failed, will retry")


# ---------------------------------------------------------------------------
# Media-follow polling loop ("follow me" playback)
# ---------------------------------------------------------------------------
# Generic-shaped like trigger_poller() (wake, find work, do it, never let
# one failure stop the rest) but not generic across skills the way triggers
# are — only one playback-capable skill (skills/music.py) exists today, so
# this calls into it directly rather than through a registered-handler
# table. If a second playback backend is ever added, this should grow a
# TriggerHandler-style registration mechanism instead of an if/elif chain;
# doing that now, for one implementation, would be speculative.

async def media_follow_poller() -> None:
    """
    Background task. Wakes every MEDIA_FOLLOW_POLL_INTERVAL_S seconds and
    moves each follow-enabled media session to wherever presence currently
    resolves its owner to.

    Runs for the lifetime of the process. Per-session errors (MPD
    unreachable, unexpected exception) are logged and skipped — one bad
    session never blocks reconciliation of the others.
    """
    log.info("Media-follow poller started (interval=%ds)", MEDIA_FOLLOW_POLL_INTERVAL_S)

    while True:
        await asyncio.sleep(MEDIA_FOLLOW_POLL_INTERVAL_S)

        for session in database.get_all_media_sessions():
            try:
                await _reconcile_media_session(session)
            except Exception:
                log.exception(
                    "media_follow_poller: failed reconciling session for %s",
                    session.user_id,
                )


async def _reconcile_media_session(session) -> None:
    """
    Decide whether one follow-enabled session needs to move, end, or be
    left alone, and act on it. Split out from media_follow_poller() so the
    per-session try/except above wraps one call, not a chunk of loop body.
    """
    location = database.get_user_location(session.user_id)

    if location is None:
        # Presence gone fully unresolved (dead tag battery, out of range
        # entirely) — don't freeze at the last known room, end the session.
        log.info("Ending media session for %s: presence unresolved", session.user_id)
        await music_skill.end_session_playback(session.current_satellite)
        database.end_media_session(session.user_id)
        return

    if location == session.current_satellite:
        return  # already following correctly, nothing to do

    existing = database.get_active_session_at(location)
    if existing is not None and existing.user_id != session.user_id:
        # An incoming session never displaces one already active at the
        # destination — suppress rather than fight over the room. The
        # moving user's session ends; it does not silently pause and wait
        # to resume later, since that would be a surprise on its own.
        log.info(
            "Suppressing follow for %s into %s: occupied by %s",
            session.user_id, location, existing.user_id,
        )
        await music_skill.end_session_playback(session.current_satellite)
        database.end_media_session(session.user_id)
        return

    try:
        await music_skill.move_session_playback(
            session.current_satellite, location, session.content_ref,
        )
    except ConnectionError:
        log.error(
            "Could not move media session for %s from %s to %s — will retry next cycle",
            session.user_id, session.current_satellite, location,
        )
        return  # bookkeeping deliberately NOT updated — retry from the old room next cycle

    database.move_media_session(session.user_id, location)


# ---------------------------------------------------------------------------
# Satellite registry helpers
# ---------------------------------------------------------------------------

_IP_TO_NAME: dict[str, str] = {ip: name for name, ip in SATELLITES.items()}

# Give skills a way to turn a satellite IP into a spoken room name (e.g.
# skills/locate.py) without importing this module, which would be
# circular — every skill is itself imported by this module via
# skills.registry. See database.py's register_satellite_names() docstring.
database.register_satellite_names(_IP_TO_NAME)


def satellite_name_from_ip(ip: str) -> str | None:
    return _IP_TO_NAME.get(ip)


def satellite_ip_from_name(name: str) -> str | None:
    return SATELLITES.get(name)


def _resolve_target_satellites(raw_targets, origin_ip: str) -> list[str]:
    """
    Turn whatever the LLM extracted for "target_satellites" into a
    deduplicated list of satellite IPs.

    raw_targets is expected to be a list of room-name strings (per the
    intent-extraction prompt), but this tolerates a bare string, a missing
    key, or garbage from a small/uncooperative model rather than raising.

    Room names are matched case-insensitively and with spaces treated the
    same as underscores, so "living room" matches "living_room" in
    SATELLITES. An unrecognised name is dropped (and logged) rather than
    causing the whole request to fail — one bad name shouldn't cancel a
    request that also named a good one.

    Defaults to [origin_ip] only when the user named no *valid* target at
    all. If they explicitly named a room, even a single one, the response
    goes only there — see voice-assistant-refactor-2026-07-02.md, Part 2,
    "Why this design, and not something else" for why origin is never
    silently added alongside an explicit target.
    """
    if isinstance(raw_targets, str):
        raw_targets = [raw_targets]
    elif not isinstance(raw_targets, list):
        raw_targets = []

    resolved: list[str] = []
    seen: set[str] = set()
    for raw in raw_targets:
        raw = str(raw).strip()
        if not raw:
            continue

        # Accept a friendly name (normalised) or, tolerantly, a raw IP if
        # the model returned one directly.
        name_key = raw.lower().replace(" ", "_")
        ip = SATELLITES.get(name_key)
        if ip is None and raw in _IP_TO_NAME:
            ip = raw

        if ip is None:
            log.warning("Unrecognised target satellite %r — ignoring", raw)
            continue

        if ip not in seen:
            seen.add(ip)
            resolved.append(ip)

    return resolved if resolved else [origin_ip]


# ---------------------------------------------------------------------------
# Per-connection handler (inbound from STT server)
# ---------------------------------------------------------------------------

async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """
    Handle one connection from the STT server.

    Expects a single Wyoming Transcript event. The satellite peer IP is in
    the event's data field (set by the STT server).

    If a clarification context is pending for this satellite, the transcript
    is routed to slot-filling rather than full intent extraction.
    """
    peer = writer.get_extra_info("peername")
    log.debug("Connection from STT server at %s", peer)

    try:
        event = await async_read_event(reader)
        if event is None:
            log.warning("STT server closed connection before sending a Transcript")
            return

        log.debug("Received event: type=%r data=%r payload=%r", event.type, event.data, event.payload)

        if not Transcript.is_type(event.type):
            log.warning("Expected Transcript event, got %r — ignoring", event.type)
            return

        transcript_event = Transcript.from_event(event)
        text = transcript_event.text.strip()
        satellite_ip = event.data["satellite_ip"]
        log.debug("satellite IP extracted, %s", satellite_ip)
        satellite_name = satellite_name_from_ip(satellite_ip) or satellite_ip

        # user_id/user_confidence come from the STT server's speaker-ID step.
        # Default to "unknown"/0.0 rather than KeyError so this still works
        # against an STT server that hasn't been upgraded to send them yet.
        user_id = event.data.get("user_id", "unknown")
        user_confidence = event.data.get("user_confidence", 0.0)

        log.info(
            "Transcript from %s (user=%r, confidence=%.3f): %r",
            satellite_name, user_id, user_confidence, text,
        )

        if not text:
            log.warning("Empty transcript from %s — nothing to do", satellite_name)
            return

        # Check for a pending clarification context before running intent extraction.
        ctx = conversation_state.get(satellite_ip)

        if ctx is not None:
            # We are mid-clarification. Try to fill the missing slot from this reply.
            log.info(
                "Clarification reply from %s (intent=%r missing_slot=%r turn=%d, "
                "owner=%r): %r",
                satellite_name, ctx.intent, ctx.missing_slot, ctx.turn, ctx.user_id, text,
            )
            if ctx.user_id != user_id:
                log.debug(
                    "Clarification answered by a different speaker (%r) than the "
                    "one who asked (%r) — action still belongs to %r",
                    user_id, ctx.user_id, ctx.user_id,
                )
            
            conversation_state.increment_turn(satellite_ip)

            try:
                fill_result = _call_ollama_slot_fill(text, ctx.intent, ctx.missing_slot)
                updated_slots = _merge_slot_from_fill(ctx.intent, ctx.missing_slot, fill_result, ctx.slots)
                log.debug("Slot fill succeeded: %s → %r", ctx.missing_slot, updated_slots.get(ctx.missing_slot))
            except (urllib.error.URLError, TimeoutError) as exc:
                log.error("Ollama unreachable during slot fill: %s", exc)
                conversation_state.clear(satellite_ip)
                await audio_io.send_canned("error", satellite_ip)
                return
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                # Model gave unusable output. If turns remain, re-ask; otherwise give up.
                log.warning("Slot fill failed for %r: %s", ctx.missing_slot, exc)
                remaining_ctx = conversation_state.get(satellite_ip)
                if remaining_ctx is not None:
                    # Turns not yet exhausted — re-ask the same question.
                    try:
                        await send_to_tts(ctx.question, satellite_ip, listen_after=True)
                    except (ConnectionError, OSError):
                        log.exception("TTS unreachable when re-asking clarification")
                        conversation_state.clear(satellite_ip)
                        await audio_io.send_canned("error", satellite_ip)
                else:
                    # Max turns hit inside get() above — context already cleared.
                    log.info("Max clarification turns reached for %s — resetting", satellite_name)
                    await audio_io.send_canned("unknown", satellite_ip)
                return

            # Slot filled successfully. Clear context and re-dispatch with completed slots.
            conversation_state.clear(satellite_ip)
            intent = {"intent": ctx.intent, "slots": updated_slots}
            await dispatch(intent, satellite_ip, ctx.target_satellites, ctx.user_id)
            return

        # No pending clarification — normal intent extraction path.
        intent = extract_intent(text)
        target_satellites = _resolve_target_satellites(
            intent.get("target_satellites", []), satellite_ip,
        )
        await dispatch(intent, satellite_ip, target_satellites, user_id)

    except (ConnectionResetError, asyncio.IncompleteReadError):
        log.warning("Connection from %s dropped unexpectedly", peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Satellite link — persistent satellite-initiated connection
# ---------------------------------------------------------------------------
# Unlike handle_connection() above (one Wyoming event per STT connection,
# opened and closed per transcript), this is long-lived: a satellite opens
# one connection at startup (see wakeword/satellite_link.py) and keeps it
# open for as long as it's running, sending one newline-delimited JSON
# message per line. Reconnection on drop is entirely the satellite's
# responsibility.

async def handle_satellite_link_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """
    Handle one persistent connection from a satellite's satellite_link
    module.

    Messages are dispatched by their "type" field. Only "presence" exists
    today; an unrecognised type is logged and ignored rather than closing
    the connection — this keeps an older orchestrator forward-compatible
    with a satellite that's been upgraded to send a newer message type
    (e.g. a future on-device screen's status), and vice versa. Likewise, a
    single malformed line is logged and skipped rather than dropping the
    whole connection over one bad message.
    """
    peer = writer.get_extra_info("peername")
    satellite_ip = peer[0] if peer else None
    log.info("Satellite link connected from %s", satellite_ip)

    try:
        while True:
            line = await reader.readline()
            if not line:
                break  # satellite closed the connection

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Malformed satellite link message from %s: %r", satellite_ip, line)
                continue

            msg_type = message.get("type") if isinstance(message, dict) else None
            if msg_type == "presence":
                _handle_presence_message(satellite_ip, message)
            else:
                log.debug(
                    "Unrecognised satellite link message type %r from %s — ignoring",
                    msg_type, satellite_ip,
                )

    except (ConnectionResetError, asyncio.IncompleteReadError):
        log.warning("Satellite link from %s dropped unexpectedly", satellite_ip)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass
        log.info("Satellite link disconnected from %s", satellite_ip)


def _handle_presence_message(satellite_ip: str, message: dict) -> None:
    """
    One BLE sighting: {"type": "presence", "user_id": ..., "rssi": ...}.

    satellite_ip comes from the connection's peer address, not the message
    body — the transport already knows this reliably, and a satellite
    self-reporting its own IP could be wrong (multiple interfaces, NAT).
    """
    user_id = message.get("user_id")
    rssi = message.get("rssi")

    if not isinstance(user_id, str) or not user_id:
        log.warning("Presence message from %s missing/invalid user_id: %r", satellite_ip, message)
        return
    if not isinstance(rssi, (int, float)):
        log.warning("Presence message from %s missing/invalid rssi: %r", satellite_ip, message)
        return

    database.update_presence(user_id, satellite_ip, int(rssi))


# ---------------------------------------------------------------------------
# Startup checks
# ---------------------------------------------------------------------------

def probe_ollama() -> None:
    """
    Verify Ollama is reachable and the intent model is available.
    Raises on failure — fail loudly at startup.
    """
    req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        body = json.loads(resp.read())

    available = [m["name"] for m in body.get("models", [])]
    if not any(m.startswith(INTENT_MODEL.split(":")[0]) for m in available):
        raise RuntimeError(
            f"Intent model {INTENT_MODEL!r} not found in Ollama. "
            f"Available: {available}. Run: ollama pull {INTENT_MODEL}"
        )

    log.info("Ollama reachable; intent model %r available", INTENT_MODEL)


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def run() -> None:
    """Load resources, probe dependencies, then serve connections indefinitely."""
    log.info("Loading pre-synthesised responses from %s...", audio_io.RESPONSES_DIR)
    audio_io.load_canned_responses()  # raises on missing files — intentional

    log.info("Probing Ollama at %s...", OLLAMA_BASE_URL)
    probe_ollama()  # raises on failure — intentional

    log.info("Building intent router...")
    if not intent_router.build(REGISTERED_SKILLS):
        log.warning(
            "Intent router unavailable — every request will use the full "
            "skill list (today's behaviour). This is not fatal; see "
            "intent_router.py's build() docstring."
        )


    log.info("Initialising database...")
    database.init()  # raises on failure — intentional

    server = await asyncio.start_server(handle_connection, host=HOST, port=PORT)
    addrs = [str(sock.getsockname()) for sock in server.sockets]
    log.info("Orchestrator listening on %s (STT)", addrs)

    satellite_link_server = await asyncio.start_server(
        handle_satellite_link_connection, host=SATELLITE_LINK_HOST, port=SATELLITE_LINK_PORT,
    )
    link_addrs = [str(sock.getsockname()) for sock in satellite_link_server.sockets]
    log.info("Satellite link listening on %s", link_addrs)

    # Start the trigger poller, plus the two presence/follow-me pollers, as
    # long-lived background tasks. Three separate tasks rather than one
    # combined loop — a bug or slow pass in one (e.g. an unreachable MPD
    # during a room move) can't delay the others.
    poller_task = asyncio.create_task(trigger_poller(), name="trigger-poller")
    presence_pruner_task = asyncio.create_task(presence_pruner(), name="presence-pruner")
    media_follow_task = asyncio.create_task(media_follow_poller(), name="media-follow-poller")
    background_tasks = (poller_task, presence_pruner_task, media_follow_task)

    try:
        async with server, satellite_link_server:
            await asyncio.gather(
                server.serve_forever(),
                satellite_link_server.serve_forever(),
            )
    finally:
        for task in background_tasks:
            task.cancel()
        for task in background_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        database.close()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        #level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    except Exception:
        log.exception("Fatal error")
        sys.exit(1)

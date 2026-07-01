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
        "slots": { ... }          # intent-specific key/value pairs
    }

Supported intents are not hard-coded here — they're whatever's registered in
skills/registry.py. Each entry in that list documents its own intent name and
slots; see skills/README.md to add a new one.

Dependencies:
    pip install wyoming
    Ollama must be running and reachable at OLLAMA_BASE_URL.
    Pre-synthesised .wav files must exist at the configured paths before startup.
"""

import asyncio
import json
import logging
import sys
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Callable

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event, async_read_event, async_write_event

import audio_io
import conversation_state
import database
from conversation_state import ClarificationNeeded
from skills.base import SlotSpec, parse_value_string
from skills.registry import REGISTERED_SKILLS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: address and port this server listens on (STT server connects here)
HOST = "127.0.0.1"
PORT = 10301

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
INTENT_MODEL = "qwen2.5:3b"

# CONFIGURE: address and port of the TTS service
TTS_HOST = "127.0.0.1"
TTS_PORT = 10302

# CONFIGURE: how often the timer poller wakes to check for due timers (seconds).
# 5 seconds gives acceptable precision without hammering the DB.
TIMER_POLL_INTERVAL = 5

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
  "slots": {}
}

Supported intents and their required slots:
"""

_INTENT_PROMPT_FOOTER = """
Rules:
- Choose exactly one intent.
- Only include slots defined for that intent.
- Do not add commentary or explanation.
"""


def _build_intent_system_prompt(skills: list) -> str:
    blocks = "\n".join(skill.prompt_block for skill in skills)
    return _INTENT_PROMPT_HEADER + "\n" + blocks + _INTENT_PROMPT_FOOTER


INTENT_SYSTEM_PROMPT = _build_intent_system_prompt(REGISTERED_SKILLS)


def _call_ollama(transcript: str) -> dict:
    """
    POST to the Ollama /api/chat endpoint and return the parsed JSON intent.

    Raises urllib.error.URLError on network failure, json.JSONDecodeError on
    bad model output, KeyError if the response shape is unexpected. All
    propagate to the caller.
    """
    payload = json.dumps({
        "model": INTENT_MODEL,
        "messages": [
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user",   "content": transcript},
        ],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
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

    Returns a dict with at minimum an "intent" key. On any failure, logs the
    error and returns {"intent": "unknown", "slots": {}}.
    """
    try:
        result = _call_ollama(transcript)
        log.debug("Intent extracted: %s", result)
        return result
    except (urllib.error.URLError, TimeoutError) as exc:
        log.error("Ollama unreachable: %s", exc)
    except json.JSONDecodeError as exc:
        log.error("Ollama returned non-JSON: %s", exc)
    except KeyError as exc:
        log.error("Unexpected Ollama response shape, missing key: %s", exc)

    return {"intent": "unknown", "slots": {}}


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


async def dispatch(intent: dict, satellite_ip: str) -> None:
    """
    Fire the acknowledgement earcon, then route to the correct skill handler.

    The ack is sent as a background task so it plays while the handler runs.
    The handler's return value (if not None) is handed off to TTS.

    If the handler raises ClarificationNeeded, the question is spoken to the
    user and the context is stored so the next utterance can fill the missing
    slot without repeating intent extraction.
    """
    intent_name = intent.get("intent", "unknown")
    slots = intent.get("slots", {})
    handler = HANDLERS.get(intent_name, HANDLERS["unknown"])

    satellite_name = satellite_name_from_ip(satellite_ip) or satellite_ip
    log.info("Dispatching intent=%r slots=%s from %s", intent_name, slots, satellite_name)

    # Fire acknowledgement immediately for known intents — runs concurrently
    # with the handler so the satellite has audio feedback while we wait on
    # the TTS. Unknown intent gets no ack; it will play its own canned
    # response instead.
    if intent_name != "unknown":
        ack_task = asyncio.create_task(
            audio_io.send_canned("acknowledged", satellite_ip),
            name=f"ack-{satellite_ip}",
        )
    else:
        ack_task = None

    try:
        response_text = await handler(slots, satellite_ip)
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
        )
        conversation_state.set(satellite_ip, ctx)

        if ack_task is not None:
            await ack_task
        try:
            # listen_after=True — satellite opens mic immediately after the
            # question finishes playing, no wake word needed for the reply.
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
        try:
            await send_to_tts(response_text, satellite_ip)
        except (ConnectionError, OSError):
            log.exception("TTS service unreachable or failed for intent %r", intent_name)
            await audio_io.send_canned("error", satellite_ip)

# ---------------------------------------------------------------------------
# Timer polling loop
# ---------------------------------------------------------------------------

async def timer_poller() -> None:
    """
    Background task. Wakes every TIMER_POLL_INTERVAL seconds, checks for
    due timers in the database, fires a TTS announcement to the originating
    satellite for each one, then marks them as fired.

    Runs for the lifetime of the process. Errors on individual timer
    announcements are logged and skipped — a failed TTS call does not
    stop the poller or affect other timers.
    """
    log.info("Timer poller started (interval=%ds)", TIMER_POLL_INTERVAL)

    while True:
        await asyncio.sleep(TIMER_POLL_INTERVAL)

        now = datetime.now(timezone.utc)
        due = database.get_due_timers(now)

        for row in due:
            timer_id = row["id"]
            label = row["label"]
            satellite_ip = row["satellite_id"]

            log.info("Timer fired: id=%d label=%r satellite=%s", timer_id, label, satellite_ip)

            # Mark fired immediately — if TTS fails we still don't want to
            # re-announce on the next poll cycle.
            database.mark_timer_fired(timer_id)

            announcement = f"{label.capitalize()} timer complete."
            try:
                await send_to_tts(announcement, satellite_ip)
            except (ConnectionError, OSError):
                log.exception(
                    "TTS unreachable when announcing timer id=%d label=%r to %s",
                    timer_id, label, satellite_ip,
                )
                try:
                    await audio_io.send_canned("error", satellite_ip)
                except (ConnectionError, OSError):
                    log.exception(
                        "Could not reach satellite %s for timer id=%d — announcement lost",
                        satellite_ip, timer_id,
                    )


# ---------------------------------------------------------------------------
# Satellite registry helpers
# ---------------------------------------------------------------------------

_IP_TO_NAME: dict[str, str] = {ip: name for name, ip in SATELLITES.items()}


def satellite_name_from_ip(ip: str) -> str | None:
    return _IP_TO_NAME.get(ip)


def satellite_ip_from_name(name: str) -> str | None:
    return SATELLITES.get(name)


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

        log.info("Transcript from %s: %r", satellite_name, text)

        if not text:
            log.warning("Empty transcript from %s — nothing to do", satellite_name)
            return

        # Check for a pending clarification context before running intent extraction.
        ctx = conversation_state.get(satellite_ip)

        if ctx is not None:
            # We are mid-clarification. Try to fill the missing slot from this reply.
            log.info(
                "Clarification reply from %s (intent=%r missing_slot=%r turn=%d): %r",
                satellite_name, ctx.intent, ctx.missing_slot, ctx.turn, text,
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
            await dispatch(intent, satellite_ip)
            return

        # No pending clarification — normal intent extraction path.
        intent = extract_intent(text)
        await dispatch(intent, satellite_ip)

    except (ConnectionResetError, asyncio.IncompleteReadError):
        log.warning("Connection from %s dropped unexpectedly", peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


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

    log.info("Initialising database...")
    database.init()  # raises on failure — intentional

    server = await asyncio.start_server(handle_connection, host=HOST, port=PORT)
    addrs = [str(sock.getsockname()) for sock in server.sockets]
    log.info("Orchestrator listening on %s", addrs)

    # Start the timer poller as a long-lived background task.
    poller_task = asyncio.create_task(timer_poller(), name="timer-poller")

    try:
        async with server:
            await server.serve_forever()
    finally:
        poller_task.cancel()
        try:
            await poller_task
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

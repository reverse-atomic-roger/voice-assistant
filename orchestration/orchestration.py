#!/usr/bin/env python3
"""
orchestrator.py

Voice assistant orchestration service.

Listens for Wyoming Transcript events from the STT server, extracts intent
via a small Ollama model, dispatches to the appropriate handler, then hands
the response text off to the TTS service.

For fixed/predictable responses (unknown intent, handler errors, etc.) the
audio is pre-synthesised at startup and sent directly to the satellite as raw
PCM — no TTS round-trip needed.

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

Supported intents (Phase 1):
    timer           slots: label (str), duration_seconds (int)
    list_add        slots: list_name (str), item (str)
    list_read       slots: list_name (str)
    converse        slots: (none — escalate to large model, not yet implemented)
    unknown         slots: (none — fallback)

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
import wave
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wyoming.asr import Transcript
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event, async_read_event, async_write_event

import conversation_state
from conversation_state import ClarificationNeeded
import database

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

# CONFIGURE: port that each satellite's Wyoming audio-in server listens on
SATELLITE_WYOMING_PORT = 10500

# CONFIGURE: Ollama base URL
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# CONFIGURE: small model for intent extraction
INTENT_MODEL = "qwen2.5:3b"

# CONFIGURE: address and port of the TTS service
TTS_HOST = "127.0.0.1"
TTS_PORT = 10302

# CONFIGURE: directory containing pre-synthesised response .wav files
# Generate these once with Piper:
#   echo "Sorry, I didn't understand that." | piper --model <model> --output_file responses/unknown.wav
RESPONSES_DIR = Path(__file__).parent / "responses"

# Pre-synthesised response filenames — one .wav per fixed response
RESPONSE_FILES = {
    "unknown":      "restate.wav",       # "Please restate command."
    "error":        "error.wav",         # "Unable to comply."
    "acknowledged": "acknowledged.wav",  # "Acknowledged." / two-tone chime
}

# CONFIGURE: how often the timer poller wakes to check for due timers (seconds).
# 5 seconds gives acceptable precision without hammering the DB.
TIMER_POLL_INTERVAL = 5

# Clarification timeout and max turns are configured in conversation_state.py.

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pre-synthesised audio — loaded once at startup
# ---------------------------------------------------------------------------
# Maps response key → raw PCM bytes + audio params extracted from the WAV.
# Shape: { key: (pcm_bytes, sample_rate, sample_width, channels) }

_CANNED: dict[str, tuple[bytes, int, int, int]] = {}


def _load_wav_as_pcm(path: Path) -> tuple[bytes, int, int, int]:
    """
    Read a WAV file and return (pcm_bytes, sample_rate, sample_width, channels).
    Raises if the file is missing or not a valid WAV.
    """
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        width = wf.getsampwidth()
        channels = wf.getnchannels()
        pcm = wf.readframes(wf.getnframes())
    return pcm, rate, width, channels


def load_canned_responses() -> None:
    """
    Load all pre-synthesised response WAVs into memory.
    Raises FileNotFoundError if any configured file is missing — fail at
    startup rather than silently falling back mid-conversation.
    """
    for key, filename in RESPONSE_FILES.items():
        path = RESPONSES_DIR / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Pre-synthesised response {key!r} not found at {path}. "
                f"Generate it with Piper before starting the orchestrator."
            )
        _CANNED[key] = _load_wav_as_pcm(path)
        log.debug("Loaded canned response %r from %s", key, path)

    log.info("Loaded %d pre-synthesised responses", len(_CANNED))


# ---------------------------------------------------------------------------
# Audio — send PCM to satellite
# ---------------------------------------------------------------------------

LISTEN_AFTER_EVENT_TYPE = "listen_after"


async def send_audio_to_satellite(
    pcm: bytes,
    sample_rate: int,
    sample_width: int,
    channels: int,
    satellite_ip: str,
    listen_after: bool = False,
) -> None:
    """
    Stream raw PCM bytes to a satellite's Wyoming audio-in server.
    Opens a fresh connection per call so each audio segment is self-contained.

    If listen_after is True, a listen_after event is sent before AudioStart.
    The satellite's audio_receiver will set audio_io.listen_after after
    playback, causing wakeword_stream to open the mic immediately for a
    clarification reply — no wake word needed.
    """
    uri = f"tcp://{satellite_ip}:{SATELLITE_WYOMING_PORT}"
    chunk_size = 4096  # bytes per AudioChunk

    async with AsyncClient.from_uri(uri) as client:
        if listen_after:
            await client.write_event(Event(type=LISTEN_AFTER_EVENT_TYPE, data={}))
            log.debug("Sent listen_after signal to %s", satellite_ip)

        await client.write_event(
            AudioStart(rate=sample_rate, width=sample_width, channels=channels).event()
        )
        for offset in range(0, len(pcm), chunk_size):
            await client.write_event(
                AudioChunk(
                    audio=pcm[offset : offset + chunk_size],
                    rate=sample_rate,
                    width=sample_width,
                    channels=channels,
                ).event()
            )
        await client.write_event(AudioStop().event())

    log.debug("Sent %d PCM bytes to %s (listen_after=%s)", len(pcm), satellite_ip, listen_after)


async def send_canned(key: str, satellite_ip: str) -> None:
    """Send a pre-synthesised response to a satellite by key."""
    pcm, rate, width, channels = _CANNED[key]
    await send_audio_to_satellite(pcm, rate, width, channels, satellite_ip)


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
    await send_audio_to_satellite(pcm, rate, width, channels, satellite_ip, listen_after=listen_after)


# ---------------------------------------------------------------------------
# Ollama intent extraction
# ---------------------------------------------------------------------------

INTENT_SYSTEM_PROMPT = """\
You are an intent extraction engine for a home voice assistant.

Given a voice command transcript, return ONLY a JSON object — no prose, no
markdown fences — in this exact shape:

{
  "intent": "<intent_name>",
  "slots": {}
}

Supported intents and their required slots:

  timer
    label            (string)  short description of what the timer is for
    duration_seconds (integer) total duration in seconds

  list_add
    list_name  (string) name of the list (e.g. "shopping", "todo")
    item       (string) the item to add

  list_read
    list_name  (string) name of the list to read out

  converse
    (no slots — use when the request is conversational, ambiguous, or
     requires reasoning rather than a simple action)

  unknown
    (no slots — use when the request does not match any supported intent)

Rules:
- Choose exactly one intent.
- Only include slots defined for that intent.
- Do not add commentary or explanation.
"""


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
# The model is given the slot name and a plain-English description so it
# knows what to look for. Deterministic post-processing handles all unit
# conversion (e.g. "three minutes" → 180 seconds) — we only ask the model
# for the raw value the user stated.

_SLOT_FILL_DESCRIPTIONS: dict[str, str] = {
    "duration_seconds": (
        "The duration of the timer as stated by the user. "
        "Extract hours, minutes, and seconds as separate integer fields. "
        "Return a JSON object with keys: hours (int), minutes (int), seconds (int). "
        "Use 0 for any unit not mentioned. Example: '3 minutes' → "
        '{\"hours\": 0, \"minutes\": 3, \"seconds\": 0}'
    ),
    "label": (
        "A short description of what the timer is for. "
        "Return a JSON object with key: value (string). "
        "Example: 'the pasta' → {\"value\": \"pasta\"}"
    ),
    "list_name": (
        "The name of the list. "
        "Return a JSON object with key: value (string). "
        "Example: 'the shopping list' → {\"value\": \"shopping\"}"
    ),
    "item": (
        "The item to add to the list. "
        "Return a JSON object with key: value (string). "
        "Example: 'a pint of milk' → {\"value\": \"a pint of milk\"}"
    ),
}


def _slot_fill_system_prompt(missing_slot: str) -> str:
    description = _SLOT_FILL_DESCRIPTIONS.get(
        missing_slot,
        f'Extract the value for "{missing_slot}". Return a JSON object with key: value.',
    )
    return (
        "You are a slot extraction engine for a home voice assistant.\n\n"
        "The user is responding to a clarifying question. "
        "Extract only the requested value from their reply.\n\n"
        f"What to extract: {description}\n\n"
        "Return ONLY a JSON object — no prose, no markdown fences."
    )


def _call_ollama_slot_fill(reply: str, missing_slot: str) -> dict:
    """
    Ask the LLM to extract one named slot from a short clarifying reply.

    Returns the raw parsed JSON dict. Raises on network failure, bad JSON,
    or unexpected response shape — callers handle these.
    """
    payload = json.dumps({
        "model": INTENT_MODEL,
        "messages": [
            {"role": "system", "content": _slot_fill_system_prompt(missing_slot)},
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
    missing_slot: str,
    fill_result: dict,
    existing_slots: dict,
) -> dict:
    """
    Deterministically merge a slot-fill result into the existing slots dict.

    For duration_seconds, the LLM returns {hours, minutes, seconds} separately
    so we can do the arithmetic here rather than asking the model to multiply.
    For all other slots, we expect {value: <string>}.

    Returns the updated slots dict. Raises ValueError if the fill_result is
    unusable (e.g. all zeros for a duration, or blank string for a name).
    """
    slots = dict(existing_slots)  # don't mutate the original

    if missing_slot == "duration_seconds":
        hours   = int(fill_result.get("hours",   0))
        minutes = int(fill_result.get("minutes", 0))
        seconds = int(fill_result.get("seconds", 0))
        total = hours * 3600 + minutes * 60 + seconds
        if total <= 0:
            raise ValueError(
                f"Duration slot fill returned zero or negative total: {fill_result!r}"
            )
        slots["duration_seconds"] = total
    else:
        value = str(fill_result.get("value", "")).strip()
        if not value:
            raise ValueError(
                f"Slot fill for {missing_slot!r} returned empty value: {fill_result!r}"
            )
        slots[missing_slot] = value

    return slots


# ---------------------------------------------------------------------------
# Intent handlers
# ---------------------------------------------------------------------------
# Each handler receives the slots dict and the originating satellite IP.
# Return value is a string to be handed off to TTS, or None to suppress TTS
# (e.g. when a canned response has already been sent).
# Raise freely; the dispatcher catches and sends the canned error response.
# ---------------------------------------------------------------------------

async def handle_timer(slots: dict, satellite_ip: str) -> str | None:
    label = slots.get("label", "timer")
    duration = int(slots.get("duration_seconds", 0))

    if duration <= 0:
        raise ClarificationNeeded(
            intent="timer",
            slots=slots,
            missing_slot="duration_seconds",
            question="Please specify duration.",
        )

    fires_at = datetime.now(timezone.utc) + timedelta(seconds=duration)
    database.add_timer(label=label, fires_at=fires_at, satellite_id=satellite_ip)

    # Build the spoken duration — "3 minutes", "1 minute 30 seconds", etc.
    minutes, seconds = divmod(duration, 60)
    hours, minutes = divmod(minutes, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    duration_str = ", ".join(parts)

    log.info("Timer set: label=%r duration=%ds fires_at=%s satellite=%s",
             label, duration, fires_at.isoformat(), satellite_ip)

    return f"Timer set. {duration_str} remaining."


async def handle_list_add(slots: dict, satellite_ip: str) -> str | None:
    list_name = slots.get("list_name", "").strip()
    item = slots.get("item", "").strip()

    if not item:
        raise ClarificationNeeded(
            intent="list_add",
            slots=slots,
            missing_slot="item",
            question="Please specify item.",
        )
    if not list_name:
        raise ClarificationNeeded(
            intent="list_add",
            slots=slots,
            missing_slot="list_name",
            question="Please specify list name.",
        )

    database.add_list_item(list_name=list_name, content=item)

    log.info("List item added: list=%r item=%r", list_name, item)
    return f"{item.capitalize()} added to {list_name} list."


async def handle_list_read(slots: dict, satellite_ip: str) -> str | None:
    list_name = slots.get("list_name", "").strip()

    if not list_name:
        raise ClarificationNeeded(
            intent="list_read",
            slots=slots,
            missing_slot="list_name",
            question="Please specify list name.",
        )

    items = database.get_list_items(list_name)

    log.info("List read: list=%r items=%d", list_name, len(items))

    if not items:
        return f"{list_name.capitalize()} list contains no items."

    # Spoken as: "Shopping list. Earl Grey. Milk. Bread."
    item_str = ". ".join(item.capitalize() for item in items)
    return f"{list_name.capitalize()} list. {item_str}."


async def handle_converse(slots: dict, satellite_ip: str) -> str | None:
    log.info("STUB converse: escalation to large model not yet implemented")
    # TODO: call large Ollama model with full conversation context
    return "That function is not yet available."


async def handle_unknown(slots: dict, satellite_ip: str) -> str | None:
    log.info("Unknown intent — sending canned response")
    await send_canned("unknown", satellite_ip)
    return None  # canned audio already sent, no TTS needed


HANDLERS = {
    "timer":     handle_timer,
    "list_add":  handle_list_add,
    "list_read": handle_list_read,
    "converse":  handle_converse,
    "unknown":   handle_unknown,
}


async def dispatch(intent: dict, satellite_ip: str) -> None:
    """
    Fire the acknowledgement earcon, then route to the correct handler.

    The ack is sent as a background task so it plays while the handler runs.
    The handler's return value (if not None) is handed off to TTS.

    If the handler raises ClarificationNeeded, the question is spoken to the
    user and the context is stored so the next utterance can fill the missing
    slot without repeating intent extraction.
    """
    intent_name = intent.get("intent", "unknown")
    slots = intent.get("slots", {})
    handler = HANDLERS.get(intent_name, handle_unknown)

    satellite_name = satellite_name_from_ip(satellite_ip) or satellite_ip
    log.info("Dispatching intent=%r slots=%s from %s", intent_name, slots, satellite_name)

    # Fire acknowledgement immediately for known intents — runs concurrently
    # with the handler so the satellite has audio feedback while we wait on
    # the TTS. Unknown intent gets no ack; it will play its own canned
    # response instead.
    if intent_name != "unknown":
        ack_task = asyncio.create_task(
            send_canned("acknowledged", satellite_ip),
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
            await send_canned("error", satellite_ip)
        return
    except Exception:
        log.exception("Handler for %r raised an exception", intent_name)
        if ack_task is not None:
            await ack_task
        await send_canned("error", satellite_ip)
        return

    # Ensure the ack has finished before TTS audio starts playing.
    if ack_task is not None:
        await ack_task

    if response_text is not None:
        try:
            await send_to_tts(response_text, satellite_ip)
        except (ConnectionError, OSError):
            log.exception("TTS service unreachable or failed for intent %r", intent_name)
            await send_canned("error", satellite_ip)

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
                    await send_canned("error", satellite_ip)
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
                fill_result = _call_ollama_slot_fill(text, ctx.missing_slot)
                updated_slots = _merge_slot_from_fill(ctx.missing_slot, fill_result, ctx.slots)
                log.debug("Slot fill succeeded: %s → %r", ctx.missing_slot, updated_slots.get(ctx.missing_slot))
            except (urllib.error.URLError, TimeoutError) as exc:
                log.error("Ollama unreachable during slot fill: %s", exc)
                conversation_state.clear(satellite_ip)
                await send_canned("error", satellite_ip)
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
                        await send_canned("error", satellite_ip)
                else:
                    # Max turns hit inside get() above — context already cleared.
                    log.info("Max clarification turns reached for %s — resetting", satellite_name)
                    await send_canned("unknown", satellite_ip)
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
    log.info("Loading pre-synthesised responses from %s...", RESPONSES_DIR)
    load_canned_responses()  # raises on missing files — intentional

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

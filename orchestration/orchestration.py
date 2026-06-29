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

async def send_audio_to_satellite(
    pcm: bytes,
    sample_rate: int,
    sample_width: int,
    channels: int,
    satellite_ip: str,
) -> None:
    """
    Stream raw PCM bytes to a satellite's Wyoming audio-in server.
    Opens a fresh connection per call so each audio segment is self-contained.
    """
    uri = f"tcp://{satellite_ip}:{SATELLITE_WYOMING_PORT}"
    chunk_size = 4096  # bytes per AudioChunk

    async with AsyncClient.from_uri(uri) as client:
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

    log.debug("Sent %d PCM bytes to %s", len(pcm), satellite_ip)


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


async def send_to_tts(text: str, satellite_ip: str) -> None:
    """
    Synthesise `text` via the TTS service, then deliver the result to the
    satellite once synthesis is complete.

    Audio always comes back here rather than being sent directly from the TTS
    service — this keeps orchestration in control of playback ordering against
    earcons and other audio already queued for the same satellite.
    """
    pcm, rate, width, channels = await synthesize_speech(text)
    await send_audio_to_satellite(pcm, rate, width, channels, satellite_ip)


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
    label            (string)  short description of what the timer is for,
                                omit entirely if user gives no label
    duration_seconds (integer) total duration in seconds

  list_add
    list_name  (string) name of the list (e.g. "shopping", "todo")
    items      (list[string]) one or more items to add; always a list even for a single item

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
# Intent handlers
# ---------------------------------------------------------------------------
# Each handler receives the slots dict and the originating satellite IP.
# Return value is a string to be handed off to TTS, or None to suppress TTS
# (e.g. when a canned response has already been sent).
# Raise freely; the dispatcher catches and sends the canned error response.
# ---------------------------------------------------------------------------

_GENERIC_TIMER_LABEL = "timer"

def _duration_str(seconds: int) -> str:
    """
    Build a natural spoken duration string from a total number of seconds.
    e.g. 90 → "1 minute, 30 seconds", 3600 → "1 hour", 3661 → "1 hour, 1 minute, 1 second"
    """
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if secs:
        parts.append(f"{secs} second{'s' if secs != 1 else ''}")
    return ", ".join(parts)

async def handle_timer(slots: dict, satellite_ip: str) -> str | None:
    raw_label = slots.get("label", "").strip()
    duration = int(slots.get("duration_seconds", 0))

    # No label and no duration — nothing useful to work with.
    if not raw_label and duration <= 0:
        return "Please restate command."

    if duration <= 0:
        return "Duration not recognised. Please restate command."

    label = raw_label if raw_label else _GENERIC_TIMER_LABEL
    fires_at = datetime.now(timezone.utc) + timedelta(seconds=duration)
 
    database.add_timer(
        label=label,
        fires_at=fires_at,
        satellite_id=satellite_ip,
        duration_label=_duration_str(duration),
    )

    log.info("Timer set: label=%r duration=%ds fires_at=%s satellite=%s",
             label, duration, fires_at.isoformat(), satellite_ip)

    return f"Timer set. {_duration_str(duration)} remaining."


async def handle_list_add(slots: dict, satellite_ip: str) -> str | None:
    list_name = slots.get("list_name", "").strip()
    # Accept either a list of items or a single string for robustness against
    # models that don't always follow the schema exactly.
    raw = slots.get("items", slots.get("item", ""))
    if isinstance(raw, str):
        items = [raw.strip()] if raw.strip() else []
    else:
        items = [str(i).strip() for i in raw if str(i).strip()]

    if not items:
        return "Item not recognised. Please restate command."
    if not list_name:
        return "List name not recognised. Please restate command."

    for item in items:
        database.add_list_item(list_name=list_name, content=item)

    log.info("List items added: list=%r items=%r", list_name, items)

    # Build natural spoken confirmation.
    if len(items) == 1:
        added_str = items[0]
    elif len(items) == 2:
        added_str = f"{items[0]} and {items[1]}"
    else:
        added_str = ", ".join(i for i in items[:-1]) + f", and {items[-1]}"

    return f"{added_str} added to {list_name} list."


async def handle_list_read(slots: dict, satellite_ip: str) -> str | None:
    list_name = slots.get("list_name", "").strip()

    if not list_name:
        return "List name not recognised. Please restate command."

    items = database.get_list_items(list_name)

    log.info("List read: list=%r items=%d", list_name, len(items))

    if not items:
        return f"{list_name} list contains no items."

    # Spoken as: "Shopping list. Earl Grey. Milk. Bread."
    item_str = ". ".join(item for item in items)
    return f"{list_name} list. {item_str}."


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
            satellite_ip = row["satellite_id"]

            log.info("Timer fired: id=%d label=%r satellite=%s", timer_id, row["label"], satellite_ip)

            # Mark fired immediately — if TTS fails we still don't want to
            # re-announce on the next poll cycle.
            database.mark_timer_fired(timer_id)

                        # Build announcement. Labelled timers use the label; unlabelled
            # timers fall back to their stored duration ("3 minute timer
            # complete."). duration_seconds is NULL for labelled timers.
            label = row["label"]
            duration_label = row["duration_label"]

            if label != _GENERIC_TIMER_LABEL:
                announcement = f"{label} timer complete."
            elif duration_label:
                announcement = f"{duration_label} timer complete."
            else:
                announcement = "Timer complete."
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

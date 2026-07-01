#!/usr/bin/env python3
"""
audio_io.py

Everything involved in getting PCM audio bytes onto a satellite's speaker:
pre-synthesised "canned" responses, and the raw Wyoming audio-streaming
primitive both canned responses and live TTS output are sent through.

Split out from orchestration.py so skill handlers can send a canned
response without importing orchestration.py itself (skills/unknown.py is
the one built-in example). orchestration.py imports skills.registry at
startup to build its dispatch table, so the reverse import would be
circular — this module sits below both, with no dependency on either.
"""

import logging
import wave
from pathlib import Path

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient
from wyoming.event import Event

log = logging.getLogger(__name__)

# CONFIGURE: port that each satellite's Wyoming audio-in server listens on
SATELLITE_WYOMING_PORT = 10500

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

LISTEN_AFTER_EVENT_TYPE = "listen_after"

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

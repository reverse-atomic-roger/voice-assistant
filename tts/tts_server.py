#!/usr/bin/env python3
"""
tts_server.py

Wyoming-protocol TTS server. Listens for incoming connections from the
orchestration service, synthesises the requested text with Piper, and
streams the resulting PCM audio back down the same connection.

This server does NOT talk to satellites directly. Audio always goes back to
orchestration first, so orchestration retains full control over playback
timing (e.g. not colliding a TTS response with an earcon or music already
playing on the same satellite).

Wyoming event flow (inbound from orchestration):
    Custom event, type="synthesize", data={"text": "<text to speak>"}

Wyoming event flow (outbound to orchestration):
    AudioStart  — declares sample rate / width / channels
    AudioChunk* — raw PCM bytes (one or many)
    AudioStop   — signals end of audio

Dependencies:
    pip install wyoming piper-tts
"""

import asyncio
import logging
import sys

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import Event, async_read_event, async_write_event

from piper import PiperVoice, SynthesisConfig

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: address and port this server listens on (orchestration connects here)
HOST = "127.0.0.1"
PORT = 10302

# CONFIGURE: path to the Piper voice model (.onnx) — pick a voice that fits
# the TNG-computer feel. Download with: python -m piper.download_voices <name>
VOICE_MODEL_PATH = "en_US-fedcomp-medium.onnx"

# CONFIGURE: synthesis tuning. Defaults are Piper's; tweak to taste.
SYNTHESIS_CONFIG = SynthesisConfig(
    volume=1.0,
    length_scale=1.0,
    noise_scale=0.667,
    noise_w_scale=0.8,
    normalize_audio=True,
)

# ---------------------------------------------------------------------------
# Fixed audio constants — Piper's output format
# ---------------------------------------------------------------------------

SAMPLE_WIDTH = 2  # bytes (int16)
CHANNELS = 1

# Wyoming event type expected on the inbound connection
SYNTHESIZE_EVENT_TYPE = "synthesize"

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-connection handler
# ---------------------------------------------------------------------------

async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    voice: PiperVoice,
) -> None:
    """
    Handle one connection from orchestration: read a single "synthesize"
    event, run Piper, stream the result back as AudioStart/AudioChunk*/
    AudioStop, then close.
    """
    peer = writer.get_extra_info("peername")
    log.debug("Connection from %s", peer)

    try:
        event = await async_read_event(reader)
        if event is None:
            log.warning("Connection from %s closed before sending a request", peer)
            return

        if event.type != SYNTHESIZE_EVENT_TYPE:
            log.warning("Expected %r event, got %r — ignoring", SYNTHESIZE_EVENT_TYPE, event.type)
            return

        text = event.data["text"]
        log.info("Synthesizing for %s: %r", peer, text)

        chunk_count = 0
        sample_rate = voice.config.sample_rate

        await async_write_event(
            AudioStart(rate=sample_rate, width=SAMPLE_WIDTH, channels=CHANNELS).event(),
            writer,
        )

        for audio_chunk in voice.synthesize(text, syn_config=SYNTHESIS_CONFIG):
            await async_write_event(
                AudioChunk(
                    audio=audio_chunk.audio_int16_bytes,
                    rate=sample_rate,
                    width=SAMPLE_WIDTH,
                    channels=CHANNELS,
                ).event(),
                writer,
            )
            chunk_count += 1

        await async_write_event(AudioStop().event(), writer)

        log.debug("Sent %d audio chunks to %s", chunk_count, peer)

    except KeyError as exc:
        log.error("Malformed synthesize request from %s, missing key: %s", peer, exc)
    except (ConnectionResetError, asyncio.IncompleteReadError):
        log.warning("Connection from %s dropped unexpectedly", peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Server entry point
# ---------------------------------------------------------------------------

async def run(voice: PiperVoice) -> None:
    """Start the TCP server and serve connections indefinitely."""
    server = await asyncio.start_server(
        lambda r, w: handle_connection(r, w, voice),
        host=HOST,
        port=PORT,
    )
    addrs = [str(sock.getsockname()) for sock in server.sockets]
    log.info("TTS server listening on %s", addrs)

    async with server:
        await server.serve_forever()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Loading Piper voice from %s...", VOICE_MODEL_PATH)
    voice = PiperVoice.load(VOICE_MODEL_PATH)
    log.info("Voice loaded (sample_rate=%d).", voice.config.sample_rate)

    try:
        asyncio.run(run(voice))
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    except Exception:
        log.exception("Fatal error")
        sys.exit(1)

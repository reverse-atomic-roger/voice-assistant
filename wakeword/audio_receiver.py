#!/usr/bin/env python3
"""
audio_receiver.py

Wyoming-protocol audio server. Listens for incoming connections from the
orchestration service, receives AudioStart -> AudioChunk* -> AudioStop, and
plays the resulting PCM through the satellite's audio output via audio_io.

This is the satellite-side counterpart to orchestration.py's
send_audio_to_satellite() — same event sequence, reversed direction.

Used today for canned responses (e.g. "unknown", "acknowledged", "error").
Will carry synthesised TTS audio once the TTS service is wired up — no
change needed here, since this server only cares about the Wyoming audio
events, not where the PCM originated.

Wyoming event flow (inbound):
    AudioStart  — declares sample rate / width / channels
    AudioChunk  — raw PCM bytes (one or many)
    AudioStop   — signals end of clip; playback finishes here

Dependencies:
    pip install wyoming
"""

import asyncio
import logging

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.event import async_read_event

import audio_io

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: address and port this server listens on (orchestration connects here)
HOST = "0.0.0.0"
PORT = 10500

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """
    Handle one playback connection from orchestration, start to finish.

    Accumulates PCM chunks declared by AudioStart, then plays the full
    clip via audio_io on AudioStop. audio_io.play_pcm() serialises against
    the wakeword earcons internally, so no locking is needed here.
    """
    peer = writer.get_extra_info("peername")
    log.info("Connection from %s", peer)

    audio_buffer = bytearray()
    rate = width = channels = None

    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                log.debug("Connection closed by %s", peer)
                break

            if AudioStart.is_type(event.type):
                start = AudioStart.from_event(event)
                rate, width, channels = start.rate, start.width, start.channels
                audio_buffer.clear()
                log.debug("AudioStart from %s (rate=%d width=%d channels=%d)", peer, rate, width, channels)

            elif AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                audio_buffer.extend(chunk.audio)

            elif AudioStop.is_type(event.type):
                log.debug("AudioStop from %s — %d bytes received", peer, len(audio_buffer))

                if rate is None:
                    log.warning("AudioStop from %s with no preceding AudioStart — discarding", peer)
                    continue

                await audio_io.play_pcm(bytes(audio_buffer), rate, width, channels)
                log.info("Played %d bytes of audio from %s", len(audio_buffer), peer)

    except (ConnectionResetError, asyncio.IncompleteReadError):
        log.warning("Connection from %s dropped unexpectedly", peer)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def run() -> None:
    """Start the TCP server and serve connections indefinitely."""
    server = await asyncio.start_server(handle_connection, host=HOST, port=PORT)
    addrs = [str(sock.getsockname()) for sock in server.sockets]
    log.info("Audio receiver listening on %s", addrs)

    async with server:
        await server.serve_forever()

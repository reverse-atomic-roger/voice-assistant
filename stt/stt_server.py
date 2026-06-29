#!/usr/bin/env python3
"""
stt_server.py

Wyoming-protocol STT server. Listens for incoming satellite connections,
receives AudioStart → AudioChunk* → AudioStop, transcribes the audio with
Faster-Whisper, and logs the transcript.

The transcript is held in a variable after transcription — ready to be passed
to the orchestration layer once that is wired up.

Wyoming event flow (inbound):
    AudioStart  — declares sample rate / width / channels
    AudioChunk  — raw PCM bytes (one or many)
    AudioStop   — signals end of utterance; transcription runs here

Dependencies:
    pip install wyoming faster-whisper
"""

import asyncio
import io
import logging
import sys
import wave

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.asr import Transcribe, Transcript
from wyoming.event import Event, async_read_event, async_write_event

from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: address and port this server listens on
HOST = "127.0.0.1"
PORT = 10300

# CONFIGURE: address and port of the orchestration service
ORCHESTRATOR_HOST = "127.0.0.1"
ORCHESTRATOR_PORT = 10301

# CONFIGURE: Faster-Whisper model size
# Options: "tiny", "tiny.en", "base", "base.en", "small", "small.en",
#          "medium", "medium.en", "large-v2", "large-v3"
# Smaller = faster but less accurate. "base.en" is a good starting point.
WHISPER_MODEL = "base.en"

# CONFIGURE: compute device — "cpu" or "cuda" (if you have a GPU on this machine)
WHISPER_DEVICE = "cpu"

# CONFIGURE: compute type — "int8" is fast and fine for CPU; "float16" for GPU
WHISPER_COMPUTE_TYPE = "int8"

# ---------------------------------------------------------------------------
# Fixed audio constants — must match satellite
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2   # bytes (int16)
CHANNELS = 1

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def pcm_bytes_to_wav_bytes(pcm: bytes) -> bytes:
    """
    Wrap raw int16 PCM bytes in a minimal WAV container so Faster-Whisper
    can accept them. Uses only stdlib — no soundfile or wave dependency needed.
    """
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()

async def forward_transcript(text: str, satellite_ip: str) -> None:
    """
    Send a Wyoming Transcript event to the orchestration service.

    The satellite's peer IP is carried in the event's data field so the
    orchestrator knows where to send the TTS response.
    """
    from wyoming.asr import Transcript
    from wyoming.event import Event, async_write_event

    reader, writer = await asyncio.open_connection(ORCHESTRATOR_HOST, ORCHESTRATOR_PORT)
    try:
        transcript_event = Transcript(text=text).event()
        event_with_ip = Event(
            type=transcript_event.type,
            data = {"text":text, "satellite_ip":satellite_ip},
            payload=transcript_event.payload,
        )
        await async_write_event(event_with_ip, writer)
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()

    log.debug("Transcript forwarded to orchestrator: %r (satellite=%s)", text, satellite_ip)

# ---------------------------------------------------------------------------
# Per-connection handler
# ---------------------------------------------------------------------------

async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    model: WhisperModel,
) -> None:
    """
    Handle one satellite connection from open to close.

    Accumulates PCM chunks, transcribes on AudioStop, logs the result.
    The `transcript` variable is the hand-off point for orchestration.
    """
    peer = writer.get_extra_info("peername")
    log.info("Connection from %s", peer)

    audio_buffer = bytearray()
    transcript: tuple[str, tuple] | None = None  # (transcribed text, peer (IP and port))

    try:
        while True:
            event = await async_read_event(reader)
            if event is None:
                log.debug("Connection closed by %s", peer)
                break

            if AudioStart.is_type(event.type):
                audio_buffer.clear()
                log.debug("AudioStart from %s", peer)

            elif AudioChunk.is_type(event.type):
                chunk = AudioChunk.from_event(event)
                audio_buffer.extend(chunk.audio)

            elif AudioStop.is_type(event.type):
                log.debug(
                    "AudioStop from %s — %d bytes (%.2fs) received",
                    peer,
                    len(audio_buffer),
                    len(audio_buffer) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS),
                )

                # Transcribe ------------------------------------------------
                wav_bytes = pcm_bytes_to_wav_bytes(bytes(audio_buffer))
                segments, info = model.transcribe(
                    io.BytesIO(wav_bytes),
                    beam_size=5,
                    language="en",
                )
                transcript = (" ".join(seg.text.strip() for seg in segments).strip(), peer)
                # -----------------------------------------------------------

                log.info("Transcript from %s: %r", transcript[1], transcript[0])

                # TODO: pass `transcript` to orchestration layer here
                try:
                    await forward_transcript(transcript[0], str(peer[0]))
                    log.info("Forwarding transcript %s to %r", transcript[0], str(peer[0]))
                except OSError as exc:
                    log.error("Failed to forward transcript to orchestrator: %s", exc)

                # Send the Transcript event back so the Wyoming protocol
                # handshake is complete (useful for future chaining).
                await async_write_event(
                    Transcript(text=transcript[0]).event(),
                    writer,
                )

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

async def run(model: WhisperModel) -> None:
    """Start the TCP server and serve connections indefinitely."""
    server = await asyncio.start_server(
        lambda r, w: handle_connection(r, w, model),
        host=HOST,
        port=PORT,
    )
    addrs = [str(sock.getsockname()) for sock in server.sockets]
    log.info("STT server listening on %s", addrs)

    async with server:
        await server.serve_forever()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info(
        "Loading Faster-Whisper model '%s' (device=%s, compute=%s)...",
        WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE_TYPE,
    )
    model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    log.info("Model loaded.")

    try:
        asyncio.run(run(model))
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    except Exception:
        log.exception("Fatal error")
        sys.exit(1)

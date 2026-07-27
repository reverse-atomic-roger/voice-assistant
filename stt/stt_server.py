#!/usr/bin/env python3
"""
stt_server.py

Wyoming-protocol STT server. Listens for incoming satellite connections,
receives AudioStart → AudioChunk* → AudioStop, transcribes the audio with
Faster-Whisper, identifies the speaker against enrolled voice profiles, and
logs the transcript.

The transcript and identified speaker are held in variables after
processing — ready to be passed to the orchestration layer once that is
wired up.

Wyoming event flow (inbound):
    AudioStart  — declares sample rate / width / channels
    AudioChunk  — raw PCM bytes (one or many)
    AudioStop   — signals end of utterance; transcription + speaker ID run here

Voice profiles are produced separately by enroll_speaker.py and loaded from
VOICE_PROFILES_PATH at startup.

Dependencies:
    pip install wyoming faster-whisper speechbrain numpy soundfile
"""

import asyncio
import io
import json
import logging
import sys
import wave

import numpy as np
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier

from wyoming.audio import AudioChunk, AudioStart, AudioStop
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

# CONFIGURE: speaker-ID embedding model (downloaded from HuggingFace on first run)
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

# CONFIGURE: path to enrolled voice profiles, produced by enroll_speaker.py
VOICE_PROFILES_PATH = "voice_profiles.json"

# CONFIGURE: cosine-similarity threshold below which a speaker is "unknown".
# Tune this against your own household's enrolled data — run enroll_speaker.py's
# test mode and look at the gap between "self" and "sibling" scores.
SPEAKER_MATCH_THRESHOLD = 0.55

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

def load_voice_profiles(path: str) -> dict[str, np.ndarray]:
    """
    Load enrolled speaker embeddings written by enroll_speaker.py.

    Raises FileNotFoundError / json.JSONDecodeError on failure — the caller
    decides whether that's fatal (see __main__).
    """
    with open(path) as f:
        raw = json.load(f)
    return {name: np.array(vec, dtype=np.float32) for name, vec in raw.items()}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def identify_speaker(
    embedding: np.ndarray,
    profiles: dict[str, np.ndarray],
    threshold: float = SPEAKER_MATCH_THRESHOLD,
) -> tuple[str, float]:
    """
    Compare an utterance embedding against all enrolled profiles.

    Returns (user_id, confidence). user_id is "unknown" if the best match
    doesn't clear `threshold` — callers must handle that case explicitly
    rather than assuming a name is always returned.
    """
    best_name, best_score = "unknown", -1.0
    for name, ref in profiles.items():
        score = cosine_similarity(embedding, ref)
        if score > best_score:
            best_name, best_score = name, score

    if best_score < threshold:
        return "unknown", best_score
    return best_name, best_score


async def forward_transcript(
    text: str, satellite_ip: str, user_id: str, user_confidence: float
) -> None:
    """
    Send a Wyoming Transcript event to the orchestration service.

    The satellite's peer IP and the identified speaker are carried in the
    event's data field so the orchestrator knows where to send the TTS
    response and whose profile to apply.
    """
    from wyoming.asr import Transcript
    from wyoming.event import Event, async_write_event

    reader, writer = await asyncio.open_connection(ORCHESTRATOR_HOST, ORCHESTRATOR_PORT)
    try:
        transcript_event = Transcript(text=text).event()
        event_with_ip = Event(
            type=transcript_event.type,
            data={
                "text": text,
                "satellite_ip": satellite_ip,
                "user_id": user_id,
                "user_confidence": user_confidence,
            },
            payload=transcript_event.payload,
        )
        await async_write_event(event_with_ip, writer)
        await writer.drain()
    finally:
        writer.close()
        await writer.wait_closed()

    log.debug(
        "Transcript forwarded to orchestrator: %r (satellite=%s, user=%s, confidence=%.3f)",
        text, satellite_ip, user_id, user_confidence,
    )

# ---------------------------------------------------------------------------
# Per-connection handler
# ---------------------------------------------------------------------------

async def handle_connection(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    model: WhisperModel,
    speaker_model: EncoderClassifier,
    voice_profiles: dict[str, np.ndarray],
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

                # Speaker ID — reuse the same wav_bytes built for Whisper above.
                # speechbrain's EncoderClassifier.load_audio() only accepts a
                # real filesystem path (it does str(path) internally and hands
                # that to soundfile) — it does NOT accept a BytesIO, despite
                # looking like it should. Decode the WAV ourselves and hand
                # encode_batch() a tensor directly, skipping load_audio entirely.
                audio_np, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
                signal = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)
                embedding = speaker_model.encode_batch(signal).squeeze().numpy()
                user_id, user_confidence = identify_speaker(embedding, voice_profiles)
                # -----------------------------------------------------------

                log.info(
                    "Transcript from %s: %r (speaker=%s, confidence=%.3f)",
                    transcript[1], transcript[0], user_id, user_confidence,
                )

                # TODO: pass `transcript` to orchestration layer here
                try:
                    await forward_transcript(
                        transcript[0], str(peer[0]), user_id, user_confidence
                    )
                    log.info("Forwarding transcript %s to %r", transcript[0], str(peer[0]))
                except OSError as exc:
                    log.error("Failed to forward transcript to orchestrator: %s", exc)

                # No response is written back to the satellite here. Its
                # side of this connection (wakeword_stream.py) sends
                # AudioStart -> AudioChunk* -> AudioStop and closes
                # immediately after — it never reads a reply on this
                # socket. The actual reply to the user (acknowledgement,
                # TTS, etc.) is delivered separately, by the orchestrator
                # pushing audio to the satellite's own Wyoming server on
                # a different connection/port. Writing a Transcript event
                # back here used to race the satellite's close and
                # surface as a spurious "dropped unexpectedly" warning
                # below on the next read.

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

async def run(
    model: WhisperModel,
    speaker_model: EncoderClassifier,
    voice_profiles: dict[str, np.ndarray],
) -> None:
    """Start the TCP server and serve connections indefinitely."""
    server = await asyncio.start_server(
        lambda r, w: handle_connection(r, w, model, speaker_model, voice_profiles),
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

    log.info("Loading speaker-ID model '%s'...", SPEAKER_MODEL_SOURCE)
    speaker_model = EncoderClassifier.from_hparams(source=SPEAKER_MODEL_SOURCE)
    log.info("Speaker-ID model loaded.")

    log.info("Loading voice profiles from '%s'...", VOICE_PROFILES_PATH)
    try:
        voice_profiles = load_voice_profiles(VOICE_PROFILES_PATH)
    except (FileNotFoundError, json.JSONDecodeError):
        log.exception(
            "Could not load voice profiles from '%s' — run enroll_speaker.py first.",
            VOICE_PROFILES_PATH,
        )
        sys.exit(1)
    log.info("Loaded %d voice profile(s): %s", len(voice_profiles), list(voice_profiles))

    try:
        asyncio.run(run(model, speaker_model, voice_profiles))
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    except Exception:
        log.exception("Fatal error")
        sys.exit(1)

#!/usr/bin/env python3
"""
wakeword_stream.py

Listens continuously for a wake word using openwakeword, then streams
subsequent audio to a downstream Wyoming proxy over the Wyoming protocol
(AudioStart -> AudioChunk* -> AudioStop).

Audio capture ends when Silero VAD detects SILENCE_SECONDS of consecutive
silence, or MAX_STREAM_SECONDS is reached (whichever comes first).

This script does not read any response from the downstream target — the proxy
handles transcript routing from there.

Mic capture and earcon playback go through audio_io, which owns the single
PyAudio instance shared with audio_receiver.py (so the two never fight over
the same ALSA device, and playback from either source is serialised).

Clean sounds downloaded from https://www.stdimension.org/MediaLib/computere.htm

Dependencies:
    pip install openwakeword wyoming py-silero-vad-lite soundfile
"""

import asyncio
import logging
import sys
import time
import os
from pathlib import Path

import numpy as np
import openwakeword
from openwakeword.model import Model as WakeWordModel

from silero_vad_lite import SileroVAD

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncClient

import soundfile as sf

import audio_io

# ---------------------------------------------------------------------------
# Configuration — tweak these to suit your environment
# ---------------------------------------------------------------------------

# CONFIGURE: Wyoming URI of the downstream proxy
TARGET_URI = "tcp://127.0.0.1:10300"

# CONFIGURE: openwakeword model name
MODEL_DIR = Path(__file__).parent
WAKE_WORD_PATHS = [str(p) for p in MODEL_DIR.glob("*.onnx")]
WAKE_WORDS = [os.path.splitext(os.path.basename(path))[0] for path in WAKE_WORD_PATHS]

# CONFIGURE: wake word detection score threshold (0-1)
WAKE_THRESHOLD = 0.5

# CONFIGURE: seconds of consecutive silence that ends the utterance
SILENCE_SECONDS = 1.0

# CONFIGURE: hard cap on stream duration regardless of VAD
MAX_STREAM_SECONDS = 15.0

# CONFIGURE: Path and filenames for computer acknowledgement sounds
SOUND_PATH = Path(__file__).parent
START_SOUND_FILE = "voiceinput1.wav"
END_SOUND_FILE = "inputok1.wav"

# ---------------------------------------------------------------------------
# Fixed audio constants — do not change unless you change hardware
# ---------------------------------------------------------------------------

SAMPLE_RATE = audio_io.INPUT_SAMPLE_RATE   # 16000 Hz — required by openwakeword and Silero VAD
SAMPLE_WIDTH = audio_io.INPUT_SAMPLE_WIDTH  # bytes (int16)
CHANNELS = audio_io.INPUT_CHANNELS

OWW_FRAME_SAMPLES = audio_io.OWW_FRAME_SAMPLES

# VAD speech probability threshold
VAD_THRESHOLD = 0.5

# Minimum gap between activations to avoid double-triggering on the wake word tail
REFRACTORY_SECONDS = 2.0

# Log a heartbeat line at most once every this many seconds so it is easy to
# spot in the logs if the detection loop stops ticking.
HEARTBEAT_INTERVAL = 30.0

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Wyoming helpers
# ---------------------------------------------------------------------------

async def send_audio_stream(audio_frames: list[bytes]) -> None:
    """
    Open a fresh Wyoming connection, send AudioStart, all buffered frames
    as AudioChunk events, then AudioStop, and close.

    Each call opens and closes its own connection so the downstream service
    gets a clean, self-contained audio segment.
    """
    async with AsyncClient.from_uri(TARGET_URI) as client:
        await client.write_event(
            AudioStart(
                rate=SAMPLE_RATE,
                width=SAMPLE_WIDTH,
                channels=CHANNELS,
            ).event()
        )

        for frame in audio_frames:
            await client.write_event(
                AudioChunk(
                    audio=frame,
                    rate=SAMPLE_RATE,
                    width=SAMPLE_WIDTH,
                    channels=CHANNELS,
                ).event()
            )

        await client.write_event(AudioStop().event())

    log.debug("Wyoming stream sent: %d chunks", len(audio_frames))


# ---------------------------------------------------------------------------
# Earcon sounds — loaded once at startup
# ---------------------------------------------------------------------------

# (pcm_bytes, sample_rate) per earcon. Stored as float32 bytes since that's
# the format soundfile decodes to and audio_io.play_pcm() accepts directly.
_START_SOUND: tuple[bytes, int] | None = None
_END_SOUND: tuple[bytes, int] | None = None


def _load_earcon(filename: str) -> tuple[bytes, int]:
    """Read a WAV file, mix to mono if needed, return (float32 pcm bytes, rate)."""
    samples, rate = sf.read(SOUND_PATH / filename, dtype="float32")
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples.tobytes(), rate


def load_earcons() -> None:
    """Load the start/end capture earcons into memory. Call once at startup."""
    global _START_SOUND, _END_SOUND
    _START_SOUND = _load_earcon(START_SOUND_FILE)
    _END_SOUND = _load_earcon(END_SOUND_FILE)
    log.debug("Earcons loaded: start=%s end=%s", START_SOUND_FILE, END_SOUND_FILE)


# ---------------------------------------------------------------------------
# VAD-gated capture
# ---------------------------------------------------------------------------

async def _capture_and_stream(
    vad: "SileroVAD",
    silence_frames_needed: int,
    max_capture_frames: int,
    VAD_CHUNK_BYTES: int,
) -> None:
    """
    Run one VAD-gated capture cycle and stream the result to the STT server.

    Reads OWW-sized frames from the mic queue, accumulates them into 512-sample
    VAD chunks (converting int16 → float32 on the way), and stops when
    silence_frames_needed consecutive silent frames follow detected speech, or
    max_capture_frames is reached. Then sends the full capture over Wyoming.

    Plays the end earcon on success. Logs a warning and skips the earcon if
    the Wyoming send fails — the capture data is lost but the loop continues.

    Called both from the wake-word path and the prompted-listen path, so the
    start earcon is the caller's responsibility (the two paths use it
    differently).
    """
    captured: list[bytes] = []
    vad_buf = bytearray()
    consecutive_silence = 0
    speech_started = False

    for _ in range(max_capture_frames):
        frame = await audio_io.capture_queue().get()
        captured.append(frame)
        vad_buf.extend(frame)

        while len(vad_buf) >= VAD_CHUNK_BYTES:
            chunk_i16 = np.frombuffer(vad_buf[:VAD_CHUNK_BYTES], dtype=np.int16)
            del vad_buf[:VAD_CHUNK_BYTES]
            chunk_f32 = (chunk_i16.astype(np.float32) / 32768.0).tobytes()

            is_speech = vad.process(chunk_f32) >= VAD_THRESHOLD

            if is_speech:
                speech_started = True
                consecutive_silence = 0
            elif speech_started:
                consecutive_silence += 1

        # Don't start the silence countdown until speech has begun —
        # there's often a short pause between the wake word and the command.
        if speech_started and consecutive_silence >= silence_frames_needed:
            log.debug("Silence detected after %d frames — ending stream", len(captured))
            break
    else:
        log.warning(
            "Hit max stream duration (%.1fs) without silence — sending anyway",
            MAX_STREAM_SECONDS,
        )

    log.debug(
        "Captured %d frames (%.2fs), sending to %s",
        len(captured),
        len(captured) * OWW_FRAME_SAMPLES / SAMPLE_RATE,
        TARGET_URI,
    )
    try:
        await send_audio_stream(captured)
        end_pcm, end_rate = _END_SOUND
        await audio_io.play_pcm(end_pcm, end_rate, width=4, channels=1)
    except OSError as exc:
        log.error("Failed to send Wyoming stream: %s", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run() -> None:
    """Continuously detect wake word, then stream audio until silence.

    If audio_io.listen_after is set (by audio_receiver when the orchestrator
    signals it expects a reply), one capture cycle runs immediately without
    requiring a wake word. The start earcon is replaced by a softer prompt
    tone to indicate the mic is open for a follow-up.
    """

    log.info("Loading openwakeword model (%s)...", WAKE_WORDS)
    openwakeword.utils.download_models()
    oww = WakeWordModel(
        wakeword_models=WAKE_WORD_PATHS,
        inference_framework="onnx",
    )

    log.info("Loading Silero VAD...")
    vad = SileroVAD(SAMPLE_RATE)
    # process() expects exactly 512 float32 samples normalised to [-1, 1].
    # The mic gives us int16, so we accumulate raw bytes and convert per chunk.
    VAD_SAMPLES = 512
    VAD_CHUNK_BYTES = VAD_SAMPLES * SAMPLE_WIDTH  # 1024 bytes of int16 per chunk
    vad_frame_seconds = VAD_SAMPLES / SAMPLE_RATE  # ~32 ms per VAD frame

    # Pre-compute loop bounds from config
    silence_frames_needed = int(SILENCE_SECONDS / vad_frame_seconds)
    max_capture_frames = int(MAX_STREAM_SECONDS * SAMPLE_RATE / OWW_FRAME_SAMPLES)

    load_earcons()

    log.info(
        "Listening for '%s' (wake_threshold=%.2f, silence=%.1fs, max=%.1fs, target=%s)",
        WAKE_WORDS, WAKE_THRESHOLD, SILENCE_SECONDS, MAX_STREAM_SECONDS, TARGET_URI,
    )

    # initialise variable to avoid overlapping wake word detections
    last_detection_time: float = 0.0
    _last_heartbeat: float = 0.0

    while True:
        # --- prompted listen check ---------------------------------------
        # Check before the wake word phase. If the orchestrator has asked a
        # clarifying question and set listen_after, skip straight to capture.
        # Clear the event immediately so a slow user doesn't trigger twice.
        if audio_io.listen_after.is_set():
            audio_io.listen_after.clear()
            log.info("Prompted listen — capturing reply without wake word")

            # Use the start earcon so the user knows the mic is open.
            # (Same sound as wake word activation — consistent UX.)
            start_pcm, start_rate = _START_SOUND
            await audio_io.play_pcm(start_pcm, start_rate, width=4, channels=1)

            await _capture_and_stream(vad, silence_frames_needed, max_capture_frames, VAD_CHUNK_BYTES)

            last_detection_time = time.monotonic()
            log.info("Prompted capture sent. Resuming wake word detection...")
            continue

        # --- wake word detection phase -----------------------------------
        # Heartbeat — log once per HEARTBEAT_INTERVAL so a stalled loop is
        # immediately visible in the logs.
        now_hb = time.monotonic()
        if now_hb - _last_heartbeat >= HEARTBEAT_INTERVAL:
            log.info("Wakeword loop heartbeat — listening for %s", WAKE_WORDS)
            _last_heartbeat = now_hb

        # Frames arrive from the dedicated capture thread via the queue.
        # Never blocks the event loop — if the capture thread stalls
        # (ALSA underrun, USB hiccup), the queue simply stops producing
        # and we wait here without freezing anything else.
        raw = await audio_io.capture_queue().get()
        audio_array = np.frombuffer(raw, dtype=np.int16)
        prediction = oww.predict(audio_array)

        for mdl in WAKE_WORDS:
            score = prediction.get(mdl, 0.0)
            if score >= WAKE_THRESHOLD:
                break
        else:
            continue

        now = time.monotonic()
        if (now - last_detection_time) < REFRACTORY_SECONDS:
            log.debug("Wake word suppressed (refractory period)")
            continue

        last_detection_time = now
        log.info("Wake word detected (score=%.3f) — capturing...", score)

        start_pcm, start_rate = _START_SOUND
        await audio_io.play_pcm(start_pcm, start_rate, width=4, channels=1)

        await _capture_and_stream(vad, silence_frames_needed, max_capture_frames, VAD_CHUNK_BYTES)

        last_detection_time = time.monotonic()  # reset after streaming completes, otherwise oww just reads the next frame and triggers again
        log.info("Stream sent. Listening again...")

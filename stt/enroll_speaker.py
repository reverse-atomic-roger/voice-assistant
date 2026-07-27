#!/usr/bin/env python3
"""
enroll_speaker.py

Records a handful of short clips from each household member and writes their
averaged speaker embeddings to a JSON file, for use by stt_server.py's
speaker-ID step.

Run in two modes:

    python enroll_speaker.py
        Enrollment mode (default). Walks through USERS in order, records
        PHRASES_PER_USER clips from each, and writes VOICE_PROFILES_PATH.
        Re-running overwrites any existing profile for the same name.

    python enroll_speaker.py --test
        Test mode. Records a single clip and prints its cosine similarity
        against every enrolled profile, without touching the profiles file.
        Use this to sanity-check enrollment quality and to pick a sensible
        SPEAKER_MATCH_THRESHOLD in stt_server.py — look at the gap between
        the "self" score and the next-closest sibling's score.

Dependencies:
    pip install sounddevice numpy speechbrain soundfile
"""

import argparse
import io
import json
import logging
import sys
import wave

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from speechbrain.inference.speaker import EncoderClassifier

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: household members to enrol, and the name each will be identified as
USERS = ["alice", "bob", "carol"]

# CONFIGURE: how many separate clips to record per person. More clips gives
# a more representative averaged embedding, at the cost of a longer session.
PHRASES_PER_USER = 5

# CONFIGURE: length of each recorded clip, in seconds. Long enough to say
# the wake phrase plus a few words naturally.
RECORDING_SECONDS = 3

# CONFIGURE: must match the satellite / stt_server audio format
SAMPLE_RATE = 16000
SAMPLE_WIDTH = 2  # bytes (int16)
CHANNELS = 1

# CONFIGURE: speaker-ID embedding model — must match stt_server.py
SPEAKER_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"

# CONFIGURE: where enrolled profiles are written — must match stt_server.py
VOICE_PROFILES_PATH = "voice_profiles.json"

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def record_clip(seconds: float) -> bytes:
    """
    Record `seconds` of mono int16 PCM audio from the default input device.
    Blocks until the recording finishes.
    """
    frame_count = int(seconds * SAMPLE_RATE)
    audio = sd.rec(frame_count, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="int16")
    sd.wait()
    return audio.tobytes()


def pcm_bytes_to_wav_bytes(pcm: bytes) -> bytes:
    """Wrap raw int16 PCM bytes in a minimal WAV container, stdlib only."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def embed_clip(speaker_model: EncoderClassifier, pcm: bytes) -> np.ndarray:
    """
    Run one recorded clip through the speaker-embedding model.

    speechbrain's EncoderClassifier.load_audio() only accepts a real
    filesystem path (it does str(path) internally and hands that to
    soundfile) — it does NOT accept a BytesIO despite looking like it
    should. Decode the WAV ourselves and hand encode_batch() a tensor
    directly, skipping load_audio entirely.
    """
    wav_bytes = pcm_bytes_to_wav_bytes(pcm)
    audio_np, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    signal = torch.from_numpy(audio_np).unsqueeze(0)  # (1, samples)
    embedding = speaker_model.encode_batch(signal).squeeze().numpy()
    return embedding


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------------------
# Profile storage
# ---------------------------------------------------------------------------

def load_voice_profiles(path: str) -> dict[str, np.ndarray]:
    """Load existing profiles, or return an empty dict if the file doesn't exist yet."""
    try:
        with open(path) as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    return {name: np.array(vec, dtype=np.float32) for name, vec in raw.items()}


def save_voice_profiles(path: str, profiles: dict[str, np.ndarray]) -> None:
    serialisable = {name: vec.tolist() for name, vec in profiles.items()}
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)


# ---------------------------------------------------------------------------
# Enrollment
# ---------------------------------------------------------------------------

def enroll_user(speaker_model: EncoderClassifier, name: str) -> np.ndarray:
    """
    Record PHRASES_PER_USER clips from one person and return their averaged,
    L2-normalised embedding.
    """
    print(f"\n=== Enrolling '{name}' ===")
    print(f"You'll record {PHRASES_PER_USER} clips of {RECORDING_SECONDS}s each.")
    print("Say the wake word naturally each time, e.g. 'Computer, what time is it?'")

    embeddings = []
    for i in range(1, PHRASES_PER_USER + 1):
        input(f"  Clip {i}/{PHRASES_PER_USER} — press Enter, then speak...")
        print("  Recording...")
        pcm = record_clip(RECORDING_SECONDS)
        embedding = embed_clip(speaker_model, pcm)
        embeddings.append(embedding)
        print("  Captured.")

    # Average, then re-normalise — cosine similarity only cares about
    # direction, and averaging unit vectors shrinks the magnitude.
    averaged = np.mean(embeddings, axis=0)
    averaged = averaged / np.linalg.norm(averaged)
    return averaged


def run_enrollment(speaker_model: EncoderClassifier) -> None:
    profiles = load_voice_profiles(VOICE_PROFILES_PATH)
    if profiles:
        log.info("Existing profiles found for: %s", list(profiles))

    for name in USERS:
        if name in profiles:
            answer = input(f"'{name}' is already enrolled. Re-record? [y/N] ")
            if answer.strip().lower() != "y":
                print(f"Skipping '{name}'.")
                continue
        profiles[name] = enroll_user(speaker_model, name)

    save_voice_profiles(VOICE_PROFILES_PATH, profiles)
    log.info("Wrote %d profile(s) to '%s'", len(profiles), VOICE_PROFILES_PATH)


# ---------------------------------------------------------------------------
# Test mode
# ---------------------------------------------------------------------------

def run_test(speaker_model: EncoderClassifier) -> None:
    profiles = load_voice_profiles(VOICE_PROFILES_PATH)
    if not profiles:
        log.error("No profiles found at '%s' — run enrollment first.", VOICE_PROFILES_PATH)
        sys.exit(1)

    input(f"Press Enter, then speak for {RECORDING_SECONDS}s...")
    print("Recording...")
    pcm = record_clip(RECORDING_SECONDS)
    embedding = embed_clip(speaker_model, pcm)

    scores = sorted(
        ((name, cosine_similarity(embedding, ref)) for name, ref in profiles.items()),
        key=lambda pair: pair[1],
        reverse=True,
    )

    print("\nSimilarity scores (highest first):")
    for name, score in scores:
        print(f"  {name:<15} {score:.3f}")

    if len(scores) >= 2:
        gap = scores[0][1] - scores[1][1]
        print(f"\nGap between best and second-best match: {gap:.3f}")
    print(
        "\nSet SPEAKER_MATCH_THRESHOLD in stt_server.py somewhere below the "
        "expected 'self' score but above the closest sibling's score."
    )


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test an existing enrollment instead of recording new profiles.",
    )
    args = parser.parse_args()

    log.info("Loading speaker-ID model '%s'...", SPEAKER_MODEL_SOURCE)
    speaker_model = EncoderClassifier.from_hparams(source=SPEAKER_MODEL_SOURCE)
    log.info("Model loaded.")

    try:
        if args.test:
            run_test(speaker_model)
        else:
            run_enrollment(speaker_model)
    except KeyboardInterrupt:
        log.info("Interrupted — no changes saved beyond last completed step")
    except Exception:
        log.exception("Fatal error")
        sys.exit(1)

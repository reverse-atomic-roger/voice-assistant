#!/usr/bin/env python3
"""
music_indexer.py

Standalone tool: walks skills/music.py's MUSIC_ROOT (../music, relative to
the project root), reads tags, analyses tempo/key from the actual audio,
and (re)builds the tracks / track_vectors tables that skills/music.py
reads from at runtime.

This is NOT imported by the running assistant, and deliberately lives
outside skills/ for that reason — it pulls in librosa, which drags in
numpy/scipy/numba, a heavy dependency with no reason to be installed on
every deployment just to run the assistant day to day. Run this by hand
whenever the music library changes:

    python3 music_indexer.py            # index new files only
    python3 music_indexer.py --rebuild  # wipe and reindex everything

Assumes each satellite's MPD instance is configured with a music_directory
that mirrors this same folder layout (e.g. an NFS/Samba mount, or a synced
copy) — paths stored in the tracks table are relative to MUSIC_ROOT and
handed to MPD's add() verbatim by skills/music.py, so they only resolve
correctly if MPD sees the same layout.

Key/tempo detection is a Krumhansl-Kessler correlation over averaged
chroma energy (a standard, if imperfect, key-finding technique) plus
librosa's onset-based tempo estimate — good enough for "search my library
by key/tempo" and for building a mood description to embed, not intended
as studio-grade musicological analysis.

Requires Ollama reachable with an embedding model pulled, matching
skills/music.py's EMBED_MODEL:
    ollama pull nomic-embed-text

Dependencies beyond what the assistant itself needs:
    pip install librosa mutagen
"""

import argparse
import json
import logging
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
from mutagen.flac import FLAC

import database
from skills import music  # registers schema at import time; reuses config

log = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".flac"}

# Krumhansl-Kessler key profiles (relative perceived stability of each pitch
# class within a key), used for correlation-based key detection.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot wheel codes, indexed the same way as _PITCH_CLASSES/_MAJOR_PROFILE
# above. Not read by anything yet, but essentially free to store now and
# saves a re-index later if a "harmonically compatible" playlist feature
# gets added.
_CAMELOT_MAJOR = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
_CAMELOT_MINOR = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]


def _detect_key(chroma_mean: np.ndarray) -> tuple[str, str]:
    """
    Correlate averaged chroma energy against the major and minor profiles
    at every one of the 12 rotations; return (key_name, camelot_key) for
    whichever rotation/mode correlates best.
    """
    best_score = -2.0
    best_name = "unknown"
    best_camelot = ""
    for i in range(12):
        major_score = np.corrcoef(chroma_mean, np.roll(_MAJOR_PROFILE, i))[0, 1]
        minor_score = np.corrcoef(chroma_mean, np.roll(_MINOR_PROFILE, i))[0, 1]
        if major_score > best_score:
            best_score, best_name, best_camelot = major_score, f"{_PITCH_CLASSES[i]} major", _CAMELOT_MAJOR[i]
        if minor_score > best_score:
            best_score, best_name, best_camelot = minor_score, f"{_PITCH_CLASSES[i]} minor", _CAMELOT_MINOR[i]
    return best_name, best_camelot


def _tempo_word(bpm: float) -> str:
    if bpm < 90:
        return "slow, mellow"
    if bpm < 120:
        return "moderate"
    if bpm < 150:
        return "upbeat"
    return "fast, high-energy"


def _energy_word(rms_mean: float, rms_low: float, rms_high: float) -> str:
    """
    rms_low/rms_high are the ~5th/95th percentile RMS across the current
    indexing batch — energy words are relative to this library, not an
    absolute loudness scale, since "loud" means different things across
    genres.
    """
    if rms_high <= rms_low:
        return "moderate"
    position = (rms_mean - rms_low) / (rms_high - rms_low)
    if position < 0.33:
        return "low"
    if position < 0.66:
        return "moderate"
    return "high"


def analyze_audio(path: Path) -> dict:
    """
    Load the audio and return {tempo_bpm, key_name, camelot_key,
    duration_s, rms_mean}. Raises on any librosa/file failure — the caller
    logs and skips the file rather than aborting the whole run.
    """
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration_s = librosa.get_duration(y=y, sr=sr)

    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo)[0])

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    key_name, camelot_key = _detect_key(chroma.mean(axis=1))

    rms_mean = float(librosa.feature.rms(y=y).mean())

    return {
        "tempo_bpm": tempo_bpm,
        "key_name": key_name,
        "camelot_key": camelot_key,
        "duration_s": float(duration_s),
        "rms_mean": rms_mean,
    }


def read_tags(path: Path) -> dict:
    """Best-effort tag read; missing tags fall back to sensible defaults."""
    try:
        tags = FLAC(str(path)).tags or {}
    except Exception as exc:
        log.warning("Could not read tags from %s: %s", path, exc)
        tags = {}

    def _first(key: str, default: str = "") -> str:
        values = tags.get(key)
        return values[0] if values else default

    return {
        "title": _first("title", path.stem),
        "artist": _first("artist", "Unknown Artist"),
        "album": _first("album", "Unknown Album"),
        "track_no": _first("tracknumber", ""),
        "genre": _first("genre", ""),
    }


def build_mood_text(tags: dict, features: dict, rms_low: float, rms_high: float) -> str:
    """
    Compose the natural-language description that gets embedded into
    track_vectors. This — not the raw tempo/key numbers — is what a mood
    query like "something cheerful" is actually matched against, so it
    spells out qualities in words a mood description would also use
    (bright/moody, upbeat/mellow, energy) rather than just numbers.
    """
    key_name = features["key_name"]
    mode_word = "bright, uplifting" if key_name.endswith("major") else "moody, introspective"
    tempo_word = _tempo_word(features["tempo_bpm"])
    energy_word = _energy_word(features["rms_mean"], rms_low, rms_high)
    genre_part = f" Genre: {tags['genre']}." if tags["genre"] else ""

    return (
        f"{tags['title']} by {tags['artist']}, from the album {tags['album']}."
        f"{genre_part} Tempo: {tempo_word} ({features['tempo_bpm']:.0f} BPM)."
        f" Key: {key_name} — {mode_word}. {energy_word.capitalize()} energy."
    )


def _ollama_embed(text: str) -> list[float]:
    payload = json.dumps({"model": music.EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{music.OLLAMA_BASE_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read())
    return body["embedding"]


def find_audio_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Music root not found: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS)


def _parse_track_no(raw: str) -> int | None:
    if not raw:
        return None
    try:
        return int(str(raw).split("/")[0])
    except ValueError:
        return None


def index_library(rebuild: bool = False) -> None:
    database.init()
    music.ensure_vector_store_ready()
    conn = database.get_connection()

    if rebuild:
        log.info("Rebuild requested — clearing existing tracks and vectors")
        conn.execute("DELETE FROM playlist_tracks")
        conn.execute("DELETE FROM tracks")
        conn.execute("DELETE FROM track_vectors")
        conn.commit()

    files = find_audio_files(music.MUSIC_ROOT)
    log.info("Found %d audio file(s) under %s", len(files), music.MUSIC_ROOT)

    existing_paths = {row["path"] for row in conn.execute("SELECT path FROM tracks").fetchall()}
    to_index = [f for f in files if str(f.relative_to(music.MUSIC_ROOT)) not in existing_paths]
    log.info("%d new file(s) to analyze (%d already indexed)", len(to_index), len(existing_paths))

    if not to_index:
        log.info("Nothing new to index")
        return

    # Analyze everything first so rms_low/rms_high (used for the relative
    # "low/moderate/high energy" wording) reflect this whole batch, not
    # whatever file happened to be processed first.
    analyzed = []
    for i, path in enumerate(to_index, start=1):
        rel_path = str(path.relative_to(music.MUSIC_ROOT))
        log.info("[%d/%d] Analyzing %s", i, len(to_index), rel_path)
        try:
            tags = read_tags(path)
            features = analyze_audio(path)
        except Exception as exc:
            log.error("Skipping %s — analysis failed: %s", rel_path, exc)
            continue
        analyzed.append((rel_path, tags, features))

    if not analyzed:
        log.warning("No files could be analyzed — nothing indexed")
        return

    rms_values = sorted(a[2]["rms_mean"] for a in analyzed)
    rms_low = rms_values[len(rms_values) // 20]
    rms_high = rms_values[-max(1, len(rms_values) // 20)]

    now = datetime.now(timezone.utc).isoformat()
    indexed_count = 0
    for rel_path, tags, features in analyzed:
        mood_text = build_mood_text(tags, features, rms_low, rms_high)
        try:
            embedding = _ollama_embed(mood_text)
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            log.error("Embedding failed for %s: %s — track stored without a mood vector", rel_path, exc)
            embedding = None

        cur = conn.execute(
            "INSERT INTO tracks (path, title, artist, album, track_no, duration_s, "
            "tempo_bpm, key_name, camelot_key, mood_text, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rel_path, tags["title"], tags["artist"], tags["album"],
                _parse_track_no(tags["track_no"]), features["duration_s"],
                features["tempo_bpm"], features["key_name"], features["camelot_key"],
                mood_text, now,
            ),
        )
        track_id = cur.lastrowid
        conn.commit()

        if embedding is not None:
            conn.execute(
                "INSERT INTO track_vectors (track_id, embedding) VALUES (?, ?)",
                (track_id, music.sqlite_vec.serialize_float32(embedding)),
            )
            conn.commit()

        indexed_count += 1
        log.info(
            "Indexed %s — tempo=%.0f bpm key=%s", rel_path, features["tempo_bpm"], features["key_name"],
        )

    log.info("Indexing complete: %d track(s) added", indexed_count)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    parser = argparse.ArgumentParser(description="Index the local music library for the voice assistant.")
    parser.add_argument("--rebuild", action="store_true", help="Wipe and reindex the entire library")
    args = parser.parse_args()

    try:
        index_library(rebuild=args.rebuild)
    except Exception:
        log.exception("Indexing failed")
        sys.exit(1)

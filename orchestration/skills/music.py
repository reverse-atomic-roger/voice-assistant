"""
skills/music.py

Built-in "music" skill: play_music, make_playlist, play_playlist.

Owns three things end to end:
  - A track library (`tracks`), populated offline by music_indexer.py (a
    separate script — see its docstring) rather than by this module. This
    file only ever *reads* tracks; it never analyses audio or writes rows
    into the library.
  - A vector index (`track_vectors`, a sqlite-vec vec0 virtual table) over
    each track's mood_text embedding, so a request like "play something
    cheerful" can be matched by meaning rather than by name.
  - Playlists (`playlists` / `playlist_tracks`), which this module both
    reads and writes — a playlist is just a saved, ordered list of track
    ids, the same shape as skills/lists.py's lists but pointing at tracks
    instead of arbitrary strings.

Playback itself is delegated to MPD, one instance per satellite (Pi), on
the assumption each satellite's MPD `music_directory` mirrors this same
folder layout (e.g. an NFS/Samba mount of MUSIC_ROOT, or a synced copy) —
paths stored in `tracks.path` are relative to MUSIC_ROOT and handed to
MPD's add() verbatim. This module only ever talks to the MPD instance(s)
at target_satellites; it never plays audio through audio_io.py/TTS.

Search strategy for both play_music and make_playlist:
  1. Try a literal/fuzzy match against title/artist/album (_fuzzy_search
     for a single best track, or a plain substring scan for "all tracks by
     this artist/album" when building a playlist).
  2. If that finds nothing, fall back to vector search: embed the user's
     phrase with the same Ollama embedding model used at index time, and
     do a KNN lookup against track_vectors. This is what makes "something
     cheerful" work — it never matches step 1, so it falls through here.

Dependencies not needed elsewhere in the assistant:
    pip install python-mpd2 sqlite-vec
music_indexer.py additionally needs librosa + mutagen, but those are NOT
required just to run the assistant — see that script's docstring.

Every handler below now accepts `user_id` (per the standard Skill handler
signature) but none of them use it yet — playlists are still shared/
household-wide, so "play my playlist" resolves the same way regardless of
who asks. Making playlists genuinely per-person would mean adding an
owner column to `playlists`, deciding what "my playlist" means when two
people have separately made one with the same name, and deciding how an
unidentified speaker's "my" should degrade. That's a real design decision,
not a signature change, so it's deliberately not done here — this module
only takes the parameter so its handlers match every other skill's
signature.
"""

import difflib
import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import mpd
import sqlite_vec

import database
from skills.base import (
    ClarificationNeeded,
    Skill,
    SlotSpec,
    parse_value_string,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: root of the music library, shared with music_indexer.py. Paths
# stored in the `tracks` table are relative to this folder.
MUSIC_ROOT = Path(__file__).parent.parent.parent / "music"

# CONFIGURE: Ollama base URL and embedding model. Must match whatever
# music_indexer.py used to build track_vectors — a different model produces
# differently-shaped/differently-meaning vectors, silently corrupting mood
# search. Pull with: ollama pull nomic-embed-text
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768  # nomic-embed-text's output dimension

# CONFIGURE: MPD port, standard default, same on every satellite.
MPD_PORT = 6600

# A fuzzy title/artist/album match below this ratio is treated as "no match"
# rather than a bad guess — this is what lets a mood phrase like "something
# cheerful" fall through to vector search instead of being forced onto
# whatever track happens to score highest.
FUZZY_THRESHOLD = 0.45

# How many tracks a mood/vibe playlist request pulls in via vector search.
PLAYLIST_MOOD_LIMIT = 15

# How much a relative volume request ("turn it up") changes volume by, out
# of 100, when the user gave no specific amount.
DEFAULT_VOLUME_STEP = 15

# ---------------------------------------------------------------------------
# Storage — owned entirely by this skill module (see database.py's
# register_schema()/get_connection() docstrings, and skills/lists.py for
# the pattern this follows).
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE NOT NULL,  -- relative to MUSIC_ROOT; handed to MPD verbatim
    title       TEXT NOT NULL,
    artist      TEXT NOT NULL,
    album       TEXT NOT NULL,
    track_no    INTEGER,
    duration_s  REAL,
    tempo_bpm   REAL,                  -- computed by music_indexer.py, not this module
    key_name    TEXT,                  -- e.g. "C major", "A minor" — ditto
    camelot_key TEXT,                  -- e.g. "8B" — harmonic-mixing notation, unused
                                        -- today but cheap to keep for future key-matching
    mood_text   TEXT,                  -- natural-language description embedded into
                                        -- track_vectors; see music_indexer.py's
                                        -- build_mood_text()
    indexed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS playlists (
    id         INTEGER PRIMARY KEY,
    name       TEXT UNIQUE NOT NULL COLLATE NOCASE,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS playlist_tracks (
    id          INTEGER PRIMARY KEY,
    playlist_id INTEGER NOT NULL REFERENCES playlists(id),
    track_id    INTEGER NOT NULL REFERENCES tracks(id),
    position    INTEGER NOT NULL
);
"""

database.register_schema(_SCHEMA)

# track_vectors (the vec0 virtual table) is deliberately NOT part of _SCHEMA
# above. Loading the sqlite-vec extension has to happen on the connection
# before a vec0 table can be created or touched, and database.init() has no
# hook for "load this extension first" — only for plain DDL. Rather than
# growing database.py a bespoke extension-loading mechanism for one skill,
# this module creates track_vectors itself, lazily, the first time it's
# needed. See ensure_vector_store_ready().
_vec_ready = False


def ensure_vector_store_ready() -> None:
    """
    Load the sqlite-vec extension onto the shared connection and create
    track_vectors if it doesn't exist yet. Safe to call repeatedly — only
    does real work once per process. Also called by music_indexer.py
    before it writes embeddings, since that script uses its own connection
    to the same database file.
    """
    global _vec_ready
    if _vec_ready:
        return

    conn = database.get_connection()
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS track_vectors USING vec0("
        f"track_id INTEGER PRIMARY KEY, embedding FLOAT[{EMBED_DIM}])"
    )
    conn.commit()
    _vec_ready = True
    log.info("sqlite-vec loaded, track_vectors ready (dim=%d)", EMBED_DIM)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio() if a and b else 0.0


def _fuzzy_search(query: str, artist: str = "", album: str = "", limit: int = 1) -> list:
    """
    Rank every track in the library against `query` (and, if given, `artist`
    /`album`) using string similarity, and return the top `limit` rows whose
    score clears FUZZY_THRESHOLD.

    Deliberately loads the whole (title, artist, album) table into Python
    rather than doing this in SQL — fine for a personal library's scale, and
    it lets a mood phrase that matches nothing well fall through to
    _vector_search rather than forcing a bad literal match.
    """
    conn = database.get_connection()
    rows = conn.execute("SELECT id, path, title, artist, album FROM tracks").fetchall()

    q = (query or "").strip()
    artist = (artist or "").strip()
    album = (album or "").strip()

    scored = []
    for row in rows:
        title, row_artist, row_album = row["title"], row["artist"], row["album"]

        score = max(
            _ratio(q, title),
            _ratio(q, row_artist),
            _ratio(q, row_album),
            _ratio(q, f"{row_artist} {title}"),
            _ratio(q, f"{title} {row_artist}"),
        )
        if artist:
            score = max(score, _ratio(artist, row_artist))
        if album:
            score = max(score, _ratio(album, row_album))

        # Substring containment is a stronger signal than a fuzzy ratio —
        # boost it so "play bohemian" beats a coincidentally-similar title.
        if q and q.lower() in title.lower():
            score = max(score, 0.95)
        if artist and artist.lower() in row_artist.lower():
            score = max(score, 0.9)
        if album and album.lower() in row_album.lower():
            score = max(score, 0.9)

        scored.append((score, row))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [row for score, row in scored[:limit] if score >= FUZZY_THRESHOLD]


def _tracks_by_artist_or_album(query: str) -> list:
    """
    Every track whose artist or album contains `query` (case-insensitive
    substring). Used for playlist-building, where "all of it" is wanted
    rather than a single best match — e.g. "make a playlist of Fleetwood
    Mac songs" should pull in the whole catalogue, not the one track
    _fuzzy_search would pick as the single best guess.
    """
    conn = database.get_connection()
    like = f"%{query.strip()}%"
    return conn.execute(
        "SELECT id, path, title, artist, album FROM tracks "
        "WHERE artist LIKE ? OR album LIKE ? "
        "ORDER BY artist, album, track_no",
        (like, like),
    ).fetchall()


def _ollama_embed(text: str) -> list[float]:
    """
    Embed `text` with the same Ollama model used at index time. Raises
    urllib.error.URLError/TimeoutError on network failure, KeyError if the
    response shape is unexpected — callers treat any of these as "mood
    search unavailable" and degrade gracefully.
    """
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
    return body["embedding"]


def _vector_search(query: str, limit: int = 1) -> list:
    """
    Embed `query` and return the `limit` nearest tracks by mood_text
    embedding, nearest first. This is the "play something cheerful" path —
    only reached once literal/fuzzy matching has already come up empty.

    Raises urllib.error.URLError/TimeoutError/KeyError on embedding failure
    — callers catch these and treat mood search as unavailable rather than
    failing the whole request.
    """
    ensure_vector_store_ready()
    conn = database.get_connection()

    embedding = _ollama_embed(query)
    vec_rows = conn.execute(
        "SELECT track_id, distance FROM track_vectors "
        "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
        (sqlite_vec.serialize_float32(embedding), limit),
    ).fetchall()

    ids = [r["track_id"] for r in vec_rows]
    if not ids:
        return []

    placeholders = ",".join("?" for _ in ids)
    rows = conn.execute(
        f"SELECT id, path, title, artist, album FROM tracks WHERE id IN ({placeholders})",
        ids,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}
    # Preserve nearest-first ordering from the vector query.
    return [by_id[i] for i in ids if i in by_id]


# ---------------------------------------------------------------------------
# MPD playback
# ---------------------------------------------------------------------------
# Every MPD-touching operation (play a queue, pause, resume, change volume)
# goes through _for_each_satellite: open a connection to each target IP, run
# one action against it, close up. A failure on one target is logged and
# does not stop the others (same philosophy as
# orchestration._broadcast_response) — only raises if *every* target
# failed, so callers can tell "fully failed" apart from "worked somewhere,
# at least".

def _for_each_satellite(targets: list[str], action) -> list:
    """
    Run `action(client)` against a fresh MPD connection to each IP in
    targets. Returns the list of values action() returned, one per target
    that succeeded (skipping — and logging — any that failed). Raises
    ConnectionError only if every target failed.
    """
    results = []
    failed = []
    for ip in targets:
        client = mpd.MPDClient()
        client.timeout = 10
        try:
            client.connect(ip, MPD_PORT)
            try:
                results.append(action(client))
            finally:
                client.close()
                client.disconnect()
        except (mpd.MPDError, OSError) as exc:
            log.error("MPD command failed for %s: %s", ip, exc)
            failed.append(ip)

    if failed and len(failed) == len(targets):
        raise ConnectionError(f"Could not reach MPD on: {', '.join(failed)}")
    return results


def _play_paths_on_satellites(paths: list[str], targets: list[str]) -> None:
    """
    Replace each target satellite's MPD queue with `paths` and start
    playback. Each satellite gets its own queue — "play in the kitchen and
    living room" plays the same thing in both rooms independently, not
    synchronised.
    """
    def _do(client: mpd.MPDClient) -> None:
        client.clear()
        for path in paths:
            client.add(path)
        client.play()

    _for_each_satellite(targets, _do)


def _pause_on_satellites(targets: list[str]) -> None:
    _for_each_satellite(targets, lambda client: client.pause(1))


def _resume_on_satellites(targets: list[str]) -> None:
    _for_each_satellite(targets, lambda client: client.pause(0))


def _set_volume_on_satellites(level: int, targets: list[str]) -> None:
    """Set every target's volume to the same absolute level (0-100)."""
    _for_each_satellite(targets, lambda client: client.setvol(level))


def _adjust_volume_on_satellites(delta: int, targets: list[str]) -> int | None:
    """
    Apply a relative volume change independently on each target: read that
    satellite's current volume, add delta, clamp to 0-100, write it back.
    Deliberately per-target rather than "compute one new value and apply
    it everywhere" — rooms may already be sitting at different volumes.

    Returns the resulting level on the first satellite reached (for the
    spoken response — rooms may end up slightly different if they started
    at different volumes), or None if no target could be reached.
    """
    def _do(client: mpd.MPDClient) -> int:
        try:
            current = int(client.status().get("volume", 0) or 0)
        except (TypeError, ValueError):
            current = 0
        new_level = max(0, min(100, current + delta))
        client.setvol(new_level)
        return new_level

    results = _for_each_satellite(targets, _do)
    return results[0] if results else None


# ---------------------------------------------------------------------------
# Playlist persistence
# ---------------------------------------------------------------------------

def _get_or_create_playlist(name: str) -> int:
    conn = database.get_connection()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT OR IGNORE INTO playlists (name, created_at) VALUES (?, ?)",
        (name, now),
    )
    conn.commit()
    row = conn.execute("SELECT id FROM playlists WHERE name = ?", (name,)).fetchone()
    return row["id"]


def _save_playlist(name: str, track_ids: list[int]) -> None:
    """Overwrite the named playlist's contents with track_ids, in order."""
    conn = database.get_connection()
    playlist_id = _get_or_create_playlist(name)
    conn.execute("DELETE FROM playlist_tracks WHERE playlist_id = ?", (playlist_id,))
    conn.executemany(
        "INSERT INTO playlist_tracks (playlist_id, track_id, position) VALUES (?, ?, ?)",
        [(playlist_id, track_id, position) for position, track_id in enumerate(track_ids)],
    )
    conn.commit()


def _get_playlist_paths(name: str) -> list[str]:
    conn = database.get_connection()
    row = conn.execute("SELECT id FROM playlists WHERE name = ?", (name,)).fetchone()
    if row is None:
        return []
    rows = conn.execute(
        "SELECT t.path FROM playlist_tracks pt JOIN tracks t ON t.id = pt.track_id "
        "WHERE pt.playlist_id = ? ORDER BY pt.position ASC",
        (row["id"],),
    ).fetchall()
    return [r["path"] for r in rows]


# ---------------------------------------------------------------------------
# Skill definitions
# ---------------------------------------------------------------------------

PROMPT_BLOCK_PLAY_MUSIC = """\
  play_music
    query   (string) what the user wants to hear — a song title, artist,
            album, or a vibe/mood description. Pass through what they said
            verbatim; do not try to split it into parts yourself.
    artist  (string, optional) include this too if the user clearly named
            an artist separately from the rest of the request
    album   (string, optional) same idea, only if an album was clearly named

    Use this for both literal requests ("play Bohemian Rhapsody", "play
    some Fleetwood Mac", "play the Rumours album") and vibe/mood requests
    ("play something cheerful", "play something to relax to").
"""

PROMPT_BLOCK_MAKE_PLAYLIST = """\
  make_playlist
    playlist_name  (string) name for the new (or replaced) playlist
    query          (string) what should go in it — an artist, an album, or
                   a mood/vibe description. Pass through what the user said
                   verbatim.

    e.g. "make a playlist called workout with upbeat songs" ->
    {"playlist_name": "workout", "query": "upbeat songs"}
"""

PROMPT_BLOCK_PLAY_PLAYLIST = """\
  play_playlist
    playlist_name  (string) name of the playlist to play
"""

PROMPT_BLOCK_PAUSE_MUSIC = """\
  pause_music
    Use this when the user wants to pause whatever is currently playing —
    "pause the music", "pause", "hold on a second". No slots.
"""

PROMPT_BLOCK_RESUME_MUSIC = """\
  resume_music
    Use this when the user wants to continue music that was paused —
    "resume", "unpause", "keep playing", "continue the music". Only use
    this when they are NOT naming anything new to play — if they name a
    song/artist/album/mood, use play_music instead. No slots.
"""

PROMPT_BLOCK_SET_VOLUME = """\
  set_volume
    level      (integer, optional) an absolute target volume from 0 to 100,
               only if the user gave a specific number ("set the volume to
               40", "turn it up to 80 percent")
    direction  (string, optional) "up" or "down", only if the user wants a
               relative change without a specific target ("turn it up",
               "lower the volume", "make it quieter")
    amount     (integer, optional) how much to change by, only paired with
               direction, only if the user gave a specific amount ("turn
               it up by 20", "turn it down 10"). Omit if no number was given
               with the up/down request.

    Give either `level` alone (absolute) or `direction` (+ optional
    `amount`) for a relative change — never both. If the user just says
    "turn it up" with no number, omit `amount` entirely; do not invent one.
"""


async def handle_play_music(slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None:
    query = str(slots.get("query", "") or "").strip()
    artist = str(slots.get("artist", "") or "").strip()
    album = str(slots.get("album", "") or "").strip()

    if not query and not artist and not album:
        raise ClarificationNeeded(
            intent="play_music",
            slots=slots,
            missing_slot="query",
            question="What would you like me to play?",
        )

    search_text = query or f"{artist} {album}".strip()

    matches = _fuzzy_search(search_text, artist=artist, album=album, limit=1)
    track = matches[0] if matches else None

    if track is None:
        try:
            mood_matches = _vector_search(search_text, limit=1)
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            log.error("Mood search unavailable: %s", exc)
            mood_matches = []
        track = mood_matches[0] if mood_matches else None

    if track is None:
        return f"I couldn't find anything matching {search_text} in your library."

    try:
        _play_paths_on_satellites([track["path"]], target_satellites)
    except ConnectionError:
        log.error("Could not reach MPD on any of %s", target_satellites)
        return "I found that track but couldn't reach the music player."

    label = f"{track['title']} by {track['artist']}" if track["artist"] else track["title"]
    log.info("Playing track id=%d path=%r", track["id"], track["path"])
    return f"Playing {label}."


async def handle_make_playlist(slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None:
    name = str(slots.get("playlist_name", "") or "").strip()
    query = str(slots.get("query", "") or "").strip()

    if not name:
        raise ClarificationNeeded(
            intent="make_playlist",
            slots=slots,
            missing_slot="playlist_name",
            question="What should I call the playlist?",
        )
    if not query:
        raise ClarificationNeeded(
            intent="make_playlist",
            slots=slots,
            missing_slot="query",
            question="What kind of songs should go on it?",
        )

    matches = _tracks_by_artist_or_album(query)
    if not matches:
        try:
            matches = _vector_search(query, limit=PLAYLIST_MOOD_LIMIT)
        except (urllib.error.URLError, TimeoutError, KeyError) as exc:
            log.error("Mood search unavailable while building playlist: %s", exc)
            matches = []

    if not matches:
        return f"I couldn't find any tracks matching {query}."

    _save_playlist(name, [row["id"] for row in matches])
    log.info("Playlist saved: name=%r tracks=%d", name, len(matches))
    count = len(matches)
    return f"Made a playlist called {name} with {count} track{'s' if count != 1 else ''}."


async def handle_play_playlist(slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None:
    name = str(slots.get("playlist_name", "") or "").strip()

    if not name:
        raise ClarificationNeeded(
            intent="play_playlist",
            slots=slots,
            missing_slot="playlist_name",
            question="Which playlist would you like to play?",
        )

    paths = _get_playlist_paths(name)
    if not paths:
        return f"I couldn't find a playlist called {name}."

    try:
        _play_paths_on_satellites(paths, target_satellites)
    except ConnectionError:
        log.error("Could not reach MPD on any of %s", target_satellites)
        return "I found that playlist but couldn't reach the music player."

    count = len(paths)
    log.info("Playing playlist: name=%r tracks=%d", name, count)
    return f"Playing {name} playlist, {count} track{'s' if count != 1 else ''}."


async def handle_pause_music(slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None:
    try:
        _pause_on_satellites(target_satellites)
    except ConnectionError:
        log.error("Could not reach MPD on any of %s", target_satellites)
        return "I couldn't reach the music player to pause it."
    return "Paused."


async def handle_resume_music(slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None:
    try:
        _resume_on_satellites(target_satellites)
    except ConnectionError:
        log.error("Could not reach MPD on any of %s", target_satellites)
        return "I couldn't reach the music player to resume it."
    return "Resuming."


async def handle_set_volume(slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None:
    level = slots.get("level")
    direction = str(slots.get("direction", "") or "").strip().lower()
    amount = slots.get("amount")

    # A clarification re-ask targets "level" specifically (see
    # SKILL_SET_VOLUME's slot_specs), but that slot's own description
    # allows "up"/"down" as an answer too — treat that the same as an
    # explicit direction slot rather than failing to parse it as a number.
    if isinstance(level, str) and level.strip().lower() in ("up", "down"):
        direction = level.strip().lower()
        level = None

    if level is None and direction not in ("up", "down"):
        raise ClarificationNeeded(
            intent="set_volume",
            slots=slots,
            missing_slot="level",
            question="What would you like the volume set to, or should I turn it up or down?",
        )

    if level is not None:
        try:
            target_level = max(0, min(100, int(level)))
        except (TypeError, ValueError):
            return "I didn't catch a valid volume level."

        try:
            _set_volume_on_satellites(target_level, target_satellites)
        except ConnectionError:
            log.error("Could not reach MPD on any of %s", target_satellites)
            return "I couldn't reach the music player to change the volume."

        log.info("Volume set: level=%d targets=%s", target_level, target_satellites)
        return f"Volume set to {target_level} percent."

    # Relative change — direction is "up" or "down" here (checked above).
    try:
        step = abs(int(amount)) if amount not in (None, "") else DEFAULT_VOLUME_STEP
    except (TypeError, ValueError):
        step = DEFAULT_VOLUME_STEP
    delta = step if direction == "up" else -step

    try:
        new_level = _adjust_volume_on_satellites(delta, target_satellites)
    except ConnectionError:
        log.error("Could not reach MPD on any of %s", target_satellites)
        return "I couldn't reach the music player to change the volume."

    log.info("Volume adjusted: direction=%r amount=%d targets=%s", direction, step, target_satellites)
    if new_level is None:
        return f"Turned the volume {direction}."
    return f"Volume {direction} to {new_level} percent."


_QUERY_SLOT_DESCRIPTION = (
    "What the user wants to hear or include — a song, artist, album, or "
    "mood/vibe description, exactly as they said it. "
    "Return a JSON object with key: value (string). "
    'Example: "something cheerful" -> {"value": "something cheerful"}'
)

SKILL_PLAY_MUSIC = Skill(
    intent="play_music",
    prompt_block=PROMPT_BLOCK_PLAY_MUSIC,
    handler=handle_play_music,
    router_hint="Play a song, artist, album, or a mood/vibe description from the music library.",
    slot_specs={
        "query": SlotSpec(description=_QUERY_SLOT_DESCRIPTION, parse=parse_value_string),
    },
)

SKILL_MAKE_PLAYLIST = Skill(
    intent="make_playlist",
    prompt_block=PROMPT_BLOCK_MAKE_PLAYLIST,
    handler=handle_make_playlist,
    router_hint="Create a named music playlist from an artist, album, or mood/vibe description.",
    slot_specs={
        "playlist_name": SlotSpec(
            description=(
                "The name to give the playlist. "
                "Return a JSON object with key: value (string). "
                'Example: "call it workout" -> {"value": "workout"}'
            ),
            parse=parse_value_string,
        ),
        "query": SlotSpec(description=_QUERY_SLOT_DESCRIPTION, parse=parse_value_string),
    },
)

SKILL_PLAY_PLAYLIST = Skill(
    intent="play_playlist",
    prompt_block=PROMPT_BLOCK_PLAY_PLAYLIST,
    handler=handle_play_playlist,
    router_hint="Play a previously saved named music playlist.",
    slot_specs={
        "playlist_name": SlotSpec(
            description=(
                "The name of the playlist to play. "
                "Return a JSON object with key: value (string). "
                'Example: "play my workout playlist" -> {"value": "workout"}'
            ),
            parse=parse_value_string,
        ),
    },
)

# pause_music and resume_music take no slots — nothing for the user to ever
# get asked to clarify, so no slot_specs entries either (same reasoning as
# skills/timer.py's check_timer).
SKILL_PAUSE_MUSIC = Skill(
    intent="pause_music",
    prompt_block=PROMPT_BLOCK_PAUSE_MUSIC,
    handler=handle_pause_music,
    router_hint="Pause whatever music is currently playing.",
)

SKILL_RESUME_MUSIC = Skill(
    intent="resume_music",
    prompt_block=PROMPT_BLOCK_RESUME_MUSIC,
    handler=handle_resume_music,
    router_hint="Resume or unpause music that was previously paused.",
)

# set_volume's only required-ish slot is "level", and only when neither
# level nor a usable direction was extracted — see handle_set_volume. That
# ClarificationNeeded path re-asks the same open-ended question regardless
# of which was missing, so a single generic string parser is enough; no
# dedicated up/down parsing is needed here.
SKILL_SET_VOLUME = Skill(
    intent="set_volume",
    prompt_block=PROMPT_BLOCK_SET_VOLUME,
    handler=handle_set_volume,
    router_hint="Change the music playback volume, up, down, or to a specific level.",
    slot_specs={
        "level": SlotSpec(
            description=(
                "The target volume as a number from 0 to 100, or the "
                "direction to change it (up/down) if the user gave no "
                "specific number. "
                "Return a JSON object with key: value — either an integer "
                "0-100, or the string \"up\" or \"down\". "
                'Example: "turn it up" -> {"value": "up"}. '
                'Example: "set it to 50" -> {"value": 50}'
            ),
            parse=parse_value_string,
        ),
    },
)

# The registry (skills/registry.py) reads this one SKILLS list — see
# skills/lists.py for why a multi-intent module still exposes just one.
SKILLS = [
    SKILL_PLAY_MUSIC,
    SKILL_MAKE_PLAYLIST,
    SKILL_PLAY_PLAYLIST,
    SKILL_PAUSE_MUSIC,
    SKILL_RESUME_MUSIC,
    SKILL_SET_VOLUME,
]

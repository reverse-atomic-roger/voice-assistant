#!/usr/bin/env python3
"""
database.py

Owns the SQLite connection for the voice assistant. Manages connection
lifecycle plus the schema and queries that are genuinely orchestrator-core.

The one core, orchestrator-owned table is `triggers` — a generic queue of
"do something at time T" events. The orchestrator's trigger poller reads it
directly, outside of any skill handler, which is what makes it core state
rather than skill-owned persistence (contrast with skills/lists.py, which
owns its own table end to end).

This module deliberately does NOT know what a "timer" is, or what any other
skill's trigger means. A few things about a trigger are core concerns and
get real columns:
  - `fires_at`             — when, because the poller has to index/query on it.
  - `origin_satellite_ip`  — which satellite originally heard the request
    that scheduled this trigger. Kept even though it may not be where the
    announcement plays, because a skill may still need it later (e.g. an
    error notification, or an intercom-style skill wanting to reply to
    whoever spoke).
  - `user_id`              — who scheduled this trigger, identified by the
    STT server's speaker-ID step ("unknown" if unidentified). Same
    reasoning as origin_satellite_ip: ownership of a timer/reminder/etc. is
    a cross-skill concern, not something any one skill's payload should
    reinvent — and it's a precondition for any future per-person routing
    (e.g. announcing to whichever satellite the owner is actually near,
    once location tracking exists) rather than always broadcasting to
    target_satellites.
  - `target_satellites`    — where the resulting announcement should
    actually be delivered when the trigger fires (usually just the origin,
    but "set a timer in the kitchen" targets the kitchen instead). Routing
    audio to satellites is core infrastructure (orchestration.py /
    audio_io.py), not skill semantics, so both of these are real columns
    rather than being buried in the opaque payload below.
Everything else about what the trigger means is the skill's business and
lives in `payload`, a JSON blob this module stores and hands back without
ever inspecting its contents.

Skill-owned persistence (tables only a skill itself reads/writes, e.g.
shopping lists) does NOT live here either. A skill that needs its own table
calls `database.register_schema(...)` once at module import time with its
own CREATE TABLE DDL, and does its own reads/writes through
`database.get_connection()`. That keeps this file from growing a new
bespoke function every time someone adds a skill with storage needs — the
same reason skill handlers live in skills/*.py instead of orchestration.py.
See skills/README.md, "Owning your own persistence", for the pattern and
skills/lists.py for a worked example.

Connection is opened once at startup via init() and reused for the lifetime
of the process. SQLite in WAL mode handles the concurrent reads the trigger
polling loop and inbound requests both need without contention.

Thread safety: asyncio runs everything on one thread, so no locking is
needed here beyond what SQLite itself provides.

Schema notes:
  - triggers.fired is an INTEGER (0/1) rather than deleting rows on fire,
    so there is a record of past triggers for any future diagnostics.
  - All timestamps are UTC ISO 8601 strings — SQLite has no native datetime
    type, and strings sort correctly for the range queries the poller needs.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: path to the SQLite database file.
# Default: a "data" subdirectory next to this file, portable on both Windows
# and Linux. The directory is created on first run if it doesn't exist.
DB_PATH = Path(__file__).parent / "data" / "assistant.db"

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

_conn: sqlite3.Connection | None = None

# Schema chunks contributed by skill modules via register_schema(), applied
# alongside _CORE_SCHEMA when init() runs. Skills call register_schema() at
# their own import time (module load), which happens well before
# orchestration.py calls init() at startup, so ordering is never an issue.
_pending_schemas: list[str] = []

# Satellite IP -> human room name (e.g. "kitchen"), for skills that need to
# speak a room name back to the user (locate_user is the first example).
# orchestration.py owns the actual SATELLITES config and calls
# register_satellite_names() once at import time with its IP -> name
# mapping; this module just holds a read-only copy so a skill can look one
# up without importing orchestration.py (which would be circular — every
# skill module is itself imported by orchestration.py via skills.registry).
_satellite_names: dict[str, str] = {}


def register_satellite_names(ip_to_name: dict[str, str]) -> None:
    """
    Called once by orchestration.py at import time with its IP -> room-name
    mapping. Not meant to be called from a skill — this is configuration,
    not a per-request operation.
    """
    _satellite_names.clear()
    _satellite_names.update(ip_to_name)


def satellite_name(satellite_ip: str) -> str:
    """
    Human room name for a satellite IP (e.g. "kitchen"), or the IP itself
    if it's unrecognised — this is for spoken text, so it always returns
    something sayable rather than None, unlike get_user_location() and
    similar lookups that distinguish "unknown" from a real value.
    """
    return _satellite_names.get(satellite_ip, satellite_ip)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CORE_SCHEMA = """
-- Core, orchestrator-level tables only. Skill-owned tables (lists, and
-- whatever future skills need) are registered via register_schema() and
-- applied in init(), not hard-coded here. Media tables (tracks,
-- track_features, etc.) are created by the media indexer in Phase 2 and
-- are not initialised here either.

CREATE TABLE IF NOT EXISTS triggers (
    id                 INTEGER PRIMARY KEY,
    skill              TEXT    NOT NULL,               -- opaque skill identifier, matches TriggerHandler.skill_name
    trigger_key        TEXT    NOT NULL,                -- skill-local key, not interpreted by the core (lets a skill find/cancel its own trigger later)
    fires_at           TEXT    NOT NULL,                -- UTC ISO 8601
    origin_satellite_ip TEXT   NOT NULL,                -- satellite that made the original request
    user_id            TEXT    NOT NULL DEFAULT 'unknown', -- speaker who scheduled this trigger, from STT speaker-ID
    target_satellites  TEXT    NOT NULL DEFAULT '[]',   -- JSON array of IPs to announce to when this fires
    payload            TEXT    NOT NULL DEFAULT '{}',   -- opaque JSON owned by the skill
    fired              INTEGER NOT NULL DEFAULT 0
);

-- Speeds up the poller's "find due, unfired triggers" query.
CREATE INDEX IF NOT EXISTS idx_triggers_fires_at ON triggers (fires_at)
    WHERE fired = 0;

-- Presence: raw BLE sightings, one row per (user, satellite), upserted on
-- every scan report. Ground truth for _resolve_location() below — not
-- itself debounced or hysteresis-applied, so a caller should never read
-- this table directly; go through get_user_location()/get_users_in_room().
CREATE TABLE IF NOT EXISTS presence_sightings (
    user_id      TEXT    NOT NULL,
    satellite_ip TEXT    NOT NULL,
    rssi         INTEGER NOT NULL,
    seen_at      TEXT    NOT NULL,   -- UTC ISO 8601
    PRIMARY KEY (user_id, satellite_ip)
);

-- Presence: resolved room per user, after hysteresis (see _resolve_location).
-- Deliberately a separate table from presence_sightings above — resolution
-- changes rarely and is a considered decision, sightings arrive every few
-- seconds from every satellite in range. Mixing the two would mean either
-- debouncing writes into the raw table (defeats its purpose as ground
-- truth) or recomputing hysteresis from scratch on every read.
CREATE TABLE IF NOT EXISTS presence_resolved (
    user_id      TEXT    PRIMARY KEY,
    satellite_ip TEXT    NOT NULL,
    resolved_at  TEXT    NOT NULL,   -- UTC ISO 8601; when this resolution last changed
    confidence   INTEGER NOT NULL   -- margin (dB) it won by, for diagnostics only
);

-- Media sessions: bookkeeping for "follow me" playback. One row per user
-- with an active follow-enabled session — one user, one session, one room
-- at a time (see skills/music.py for the actual MPD move/fade mechanics;
-- this table only tracks whose session is where).
CREATE TABLE IF NOT EXISTS media_sessions (
    user_id           TEXT    PRIMARY KEY,
    content_ref       TEXT    NOT NULL,              -- opaque JSON, e.g. {"paths": [...], "index": N}
    current_satellite TEXT    NOT NULL,
    started_at        TEXT    NOT NULL,              -- UTC ISO 8601
    follow_enabled    INTEGER NOT NULL DEFAULT 1
);
"""


def _migrate_add_user_id_column(conn: sqlite3.Connection) -> None:
    """
    Add triggers.user_id to a database created before this column existed.

    CREATE TABLE IF NOT EXISTS in _CORE_SCHEMA only helps on a fresh
    database — it's a no-op against a triggers table that already exists on
    disk without this column. ALTER TABLE ADD COLUMN with a DEFAULT is safe
    and cheap even on a populated table, so we just check for the column
    and add it if missing, rather than versioning the whole schema.
    """
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(triggers)")}
    if "user_id" not in existing_columns:
        log.info("Migrating triggers table: adding user_id column")
        conn.execute(
            "ALTER TABLE triggers ADD COLUMN user_id TEXT NOT NULL DEFAULT 'unknown'"
        )
        conn.commit()


def register_schema(sql: str) -> None:
    """
    Register a chunk of schema DDL owned by a skill, to be applied when
    init() runs.

    Call this once at module level in a skill file that needs its own
    table(s) — see skills/lists.py for the pattern. Use CREATE TABLE IF NOT
    EXISTS / CREATE INDEX IF NOT EXISTS so re-running init() (e.g. in tests)
    stays safe.

    Must be called before database.init() — i.e. at skill import time, not
    from inside a handler. Raises if called afterwards, since nothing would
    apply it at that point.
    """
    if _conn is not None:
        raise RuntimeError(
            "register_schema() called after database.init() — schema "
            "registration must happen at skill import time, before the "
            "orchestrator calls init()."
        )
    _pending_schemas.append(sql)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init() -> None:
    """
    Open the database, enable WAL mode, and run the core schema plus every
    schema chunk skills registered via register_schema().

    Creates the data directory and database file if they don't exist.
    Raises on any failure — intended to be called once at orchestrator
    startup so a bad DB path or a bad skill schema fails loudly before any
    requests are served.
    """
    global _conn

    if _conn is not None:
        raise RuntimeError("database.init() called more than once")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    # WAL mode: readers don't block writers and vice versa. Safe for a
    # single-process asyncio service; good habit for when the media indexer
    # runs as a separate process in Phase 2.
    conn.execute("PRAGMA journal_mode=WAL")

    # Enforce foreign key constraints — SQLite disables them by default.
    conn.execute("PRAGMA foreign_keys=ON")

    conn.executescript(_CORE_SCHEMA)
    _migrate_add_user_id_column(conn)
    for schema in _pending_schemas:
        conn.executescript(schema)
    conn.commit()

    _conn = conn

    log.info(
        "Database initialised at %s (%d skill-owned schema chunk(s) applied)",
        DB_PATH, len(_pending_schemas),
    )


def close() -> None:
    """Close the database connection. Safe to call if init() was never called."""
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None
        log.info("Database connection closed")


def _db() -> sqlite3.Connection:
    """Return the open connection, raising if init() was not called."""
    if _conn is None:
        raise RuntimeError("database.init() must be called before any database operations")
    return _conn


def get_connection() -> sqlite3.Connection:
    """
    Return the shared connection for a skill's own queries.

    Skills that need their own table own its SQL the same way they own
    their prompt_block and handler — see skills/README.md, "Owning your
    own persistence". Only reach for this when a skill needs storage
    nothing else touches; state genuinely shared between skills or with
    the orchestrator itself (like triggers) still belongs as a real
    function in this file.

    Row factory is sqlite3.Row (dict-like access by column name). Commits
    are the caller's responsibility, same as every function in this file.
    """
    return _db()


# ---------------------------------------------------------------------------
# Triggers — the generic "do something at time T" queue
# ---------------------------------------------------------------------------
# Kept here rather than moved into any one skill: orchestration.py's trigger
# poller queries this table directly, outside of any skill handler, which
# makes it genuinely core orchestrator state rather than skill-owned
# persistence. No skill-specific concept (timer, alarm, whatever) appears
# anywhere in this section — that's the point.


@dataclass(frozen=True)
class Trigger:
    """
    One due (or pending) row from the triggers table, with `payload` and
    `target_satellites` already decoded from JSON so callers never touch
    the storage format directly.
    """
    id: int
    skill: str
    trigger_key: str
    fires_at: datetime
    origin_satellite_ip: str
    user_id: str
    target_satellites: list[str]
    payload: dict


def add_trigger(
    skill: str,
    trigger_key: str,
    fires_at: datetime,
    origin_satellite_ip: str,
    target_satellites: list[str] | None = None,
    payload: dict | None = None,
    user_id: str = "unknown",
) -> int:
    """
    Schedule a future event and return its row id.

    skill                — identifier matching a registered TriggerHandler.
                           skill_name (see skills/registry.py). The poller
                           uses this to look up which handler to call when
                           the trigger fires.
    trigger_key          — skill-local identifier (e.g. a label or generated
                           id). Never interpreted by the core; it exists so
                           a skill can later find or cancel its own trigger
                           without needing a bespoke lookup function added
                           here.
    fires_at             — must be a UTC-aware datetime; stored as an ISO
                           8601 string.
    origin_satellite_ip  — the satellite that made the original request.
                           Not necessarily where the announcement plays —
                           see target_satellites — but kept around for
                           anything that should always go back to whoever
                           asked (error notifications, future intercom-style
                           reply-to-sender use cases).
    target_satellites    — list of satellite IPs the announcement should be
                           delivered to when the trigger fires. Defaults to
                           [origin_satellite_ip] if omitted or empty — the
                           same "no target named, assume origin" rule
                           orchestration.py applies to immediate responses.
    payload              — arbitrary JSON-serialisable dict owned entirely by
                           the skill. Stored and returned verbatim; the core
                           never reads or interprets its contents.
    user_id              — speaker who scheduled this trigger, from the STT
                           server's speaker-ID step. "unknown" if
                           unidentified or below its confidence threshold.
                           Stored as a real column (not payload) for the
                           same reason as origin_satellite_ip: ownership is
                           a cross-skill concern, and future features (e.g.
                           routing an announcement to wherever the owner
                           currently is) need to query on it without every
                           skill agreeing on a payload key first.
                           Placed last, after the existing parameters, so
                           existing positional call sites in skills keep
                           working unchanged until you update them.
    """
    fires_at_str = fires_at.astimezone(timezone.utc).isoformat()
    resolved_targets = list(target_satellites) if target_satellites else [origin_satellite_ip]
    targets_str = json.dumps(resolved_targets)
    payload_str = json.dumps(payload or {})
    cur = _db().execute(
        "INSERT INTO triggers "
        "(skill, trigger_key, fires_at, origin_satellite_ip, target_satellites, payload, user_id, fired) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
        (skill, trigger_key, fires_at_str, origin_satellite_ip, targets_str, payload_str, user_id),
    )
    _db().commit()
    log.debug(
        "Trigger added: id=%d skill=%r trigger_key=%r fires_at=%s origin=%s user=%r targets=%s",
        cur.lastrowid, skill, trigger_key, fires_at_str, origin_satellite_ip, user_id, resolved_targets,
    )
    return cur.lastrowid

def _trigger_from_row(row: sqlite3.Row) -> Trigger:
    """
    Shared row -> Trigger decoding used by both get_due_triggers() and
    get_pending_triggers(), so the payload/target_satellites JSON-decode
    fallback logic lives in exactly one place.
    """
    try:
        payload = json.loads(row["payload"])
    except json.JSONDecodeError:
        log.error(
            "Trigger id=%d (skill=%r) has unparseable payload — treating as {}",
            row["id"], row["skill"],
        )
        payload = {}
    try:
        target_satellites = json.loads(row["target_satellites"])
        if not target_satellites:
            raise ValueError("empty")
    except (json.JSONDecodeError, ValueError):
        log.error(
            "Trigger id=%d (skill=%r) has unparseable/empty target_satellites — "
            "falling back to origin %s",
            row["id"], row["skill"], row["origin_satellite_ip"],
        )
        target_satellites = [row["origin_satellite_ip"]]
    return Trigger(
        id=row["id"],
        skill=row["skill"],
        trigger_key=row["trigger_key"],
        fires_at=datetime.fromisoformat(row["fires_at"]),
        origin_satellite_ip=row["origin_satellite_ip"],
        user_id=row["user_id"],
        target_satellites=target_satellites,
        payload=payload,
    )




def get_due_triggers(now: datetime) -> list[Trigger]:
    """
    Return all unfired triggers whose fires_at is at or before `now`, as
    Trigger objects with `payload` and `target_satellites` already decoded
    from JSON.

    `now` should be UTC-aware; compared as ISO 8601 strings (sorts correctly).
    """
    now_str = now.astimezone(timezone.utc).isoformat()
    rows = _db().execute(
        "SELECT id, skill, trigger_key, fires_at, origin_satellite_ip, user_id, target_satellites, payload "
        "FROM triggers WHERE fired = 0 AND fires_at <= ?",
        (now_str,),
    ).fetchall()
    return [_trigger_from_row(row) for row in rows]

def get_pending_triggers(skill: str) -> list[Trigger]:
    """
    Return every not-yet-fired trigger scheduled by `skill`, regardless of
    whether fires_at has passed yet — unlike get_due_triggers(), which is
    only for the poller's "what's ready to fire right now" sweep. This is
    for a skill asking "what's still pending for me?" (e.g. timer.py's
    check_timer intent querying its own outstanding timers).

    Ordered by fires_at ascending, so the soonest-to-fire trigger is first.
    """
    rows = _db().execute(
        "SELECT id, skill, trigger_key, fires_at, origin_satellite_ip, user_id, target_satellites, payload "
        "FROM triggers WHERE fired = 0 AND skill = ? ORDER BY fires_at ASC",
        (skill,),
    ).fetchall()
    return [_trigger_from_row(row) for row in rows]

def mark_trigger_fired(trigger_id: int) -> None:
    """Mark a trigger as fired so the poller doesn't re-announce it."""
    _db().execute("UPDATE triggers SET fired = 1 WHERE id = ?", (trigger_id,))
    _db().commit()
    log.debug("Trigger %d marked fired", trigger_id)


# ---------------------------------------------------------------------------
# Presence — BLE-tag-derived room location per user
# ---------------------------------------------------------------------------
# Core for the same reason triggers are: more than one thing reads it
# outside of a single request (the locate skill, orchestration.py's
# media_follow_poller, and check_user_at_satellite()'s request-time
# mismatch guard), and this module has no idea what "locate" or "follow me"
# mean — it only knows "where is this user right now".

# CONFIGURE: a new room's RSSI must beat the current one by this many dB
# before we switch — prevents flapping right at a room boundary.
MARGIN_DB = 8

# CONFIGURE: a sighting older than this is ignored when resolving location.
STALE_SECONDS = 45

# CONFIGURE: a sighting older than this is deleted outright by
# prune_stale_presence(). Deliberately looser than STALE_SECONDS so recent
# sighting history isn't lost the instant it stops counting for resolution.
PRUNE_SECONDS = 300


def update_presence(user_id: str, satellite_ip: str, rssi: int) -> None:
    """
    Upsert one raw BLE sighting and immediately re-resolve that user's
    location. Called on every scan report a satellite forwards — there is
    no separate ingestion queue, this is the ingestion point.
    """
    conn = _db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO presence_sightings (user_id, satellite_ip, rssi, seen_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (user_id, satellite_ip)
        DO UPDATE SET rssi = excluded.rssi, seen_at = excluded.seen_at
        """,
        (user_id, satellite_ip, rssi, now),
    )
    conn.commit()
    _resolve_location(user_id)


def _resolve_location(user_id: str) -> None:
    """
    Recompute the resolved room for one user from current sightings, with
    hysteresis (MARGIN_DB) so borderline RSSI doesn't flap between rooms.
    Called synchronously from update_presence() — resolution is cheap
    enough there's no reason to defer it to a poller.
    """
    conn = _db()
    cutoff = (datetime.now(timezone.utc) - timedelta(seconds=STALE_SECONDS)).isoformat()

    rows = conn.execute(
        """
        SELECT satellite_ip, rssi FROM presence_sightings
        WHERE user_id = ? AND seen_at >= ?
        ORDER BY rssi DESC
        """,
        (user_id, cutoff),
    ).fetchall()

    if not rows:
        return  # no fresh sightings; leave the last resolution as-is

    best_ip, best_rssi = rows[0]["satellite_ip"], rows[0]["rssi"]
    current = conn.execute(
        "SELECT satellite_ip FROM presence_resolved WHERE user_id = ?",
        (user_id,),
    ).fetchone()

    if current is None:
        # First-ever resolution for this user — no prior room to weigh
        # against, so no margin check needed.
        _write_resolution(user_id, best_ip, confidence=best_rssi)
        return

    current_ip = current["satellite_ip"]
    if best_ip == current_ip:
        return  # no change

    current_row = next((r for r in rows if r["satellite_ip"] == current_ip), None)
    current_rssi = current_row["rssi"] if current_row else None

    if current_rssi is None or (best_rssi - current_rssi) >= MARGIN_DB:
        # Either the current room has no fresh sighting to defend itself
        # with, or the contender beat it by enough margin to switch.
        margin = best_rssi - (current_rssi if current_rssi is not None else best_rssi)
        _write_resolution(user_id, best_ip, confidence=margin)
    # else: contender hasn't won by enough margin yet — stay put.


def _write_resolution(user_id: str, satellite_ip: str, confidence: int) -> None:
    conn = _db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO presence_resolved (user_id, satellite_ip, resolved_at, confidence)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (user_id) DO UPDATE SET
            satellite_ip = excluded.satellite_ip,
            resolved_at = excluded.resolved_at,
            confidence = excluded.confidence
        """,
        (user_id, satellite_ip, now, confidence),
    )
    conn.commit()
    log.debug("Presence resolved: user=%r -> %s (confidence=%d dB)", user_id, satellite_ip, confidence)


def get_user_location(user_id: str) -> str | None:
    """
    Resolved room (satellite IP) for a user, or None if unknown.

    A row present here is guaranteed fresh as of the last
    prune_stale_presence() sweep — staleness is only ever evaluated there
    (see that function's docstring), so this stays a cheap, synchronous
    read with no re-derivation of freshness on every call.
    """
    row = _db().execute(
        "SELECT satellite_ip FROM presence_resolved WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return row["satellite_ip"] if row else None


def get_users_in_room(satellite_ip: str) -> list[str]:
    """Every user currently resolved to satellite_ip."""
    rows = _db().execute(
        "SELECT user_id FROM presence_resolved WHERE satellite_ip = ?",
        (satellite_ip,),
    ).fetchall()
    return [row["user_id"] for row in rows]


def prune_stale_presence() -> None:
    """
    Call periodically (see orchestration.py's presence_pruner()). Two jobs:

      1. Clear any resolved location no longer backed by a fresh sighting
         at that satellite — tag out of range, dead battery, satellite
         offline. Note resolved_at tracks when a resolution last *changed*,
         not when it was last *confirmed*, so a user standing still for an
         hour must not be judged stale by resolved_at's age — this checks
         for a live sighting instead.
      2. Garbage-collect raw sightings older than PRUNE_SECONDS.

    This is the only place staleness is evaluated; get_user_location() and
    check_user_at_satellite() never re-check it, keeping the request path
    cheap and synchronous. One consequence: a resolution can be up to one
    poller interval stale before it's cleared here — an accepted tradeoff,
    same eventual-consistency shape as the trigger poller.
    """
    conn = _db()
    now = datetime.now(timezone.utc)
    stale_cutoff = (now - timedelta(seconds=STALE_SECONDS)).isoformat()
    prune_cutoff = (now - timedelta(seconds=PRUNE_SECONDS)).isoformat()

    conn.execute(
        """
        DELETE FROM presence_resolved
        WHERE NOT EXISTS (
            SELECT 1 FROM presence_sightings ps
            WHERE ps.user_id = presence_resolved.user_id
              AND ps.satellite_ip = presence_resolved.satellite_ip
              AND ps.seen_at >= ?
        )
        """,
        (stale_cutoff,),
    )
    conn.execute("DELETE FROM presence_sightings WHERE seen_at < ?", (prune_cutoff,))
    conn.commit()


@dataclass(frozen=True)
class PresenceCheck:
    """
    Result of cross-checking a claimed speaker identity against
    independent presence data at request time.

    trusted=False never blocks the request itself — see
    check_user_at_satellite()'s docstring — it only tells a caller whether
    user_id is reliable enough to bind identity-sensitive behaviour to
    (e.g. a follow-me media session).
    """
    trusted: bool
    resolved_location: str | None
    reason: str  # diagnostic only — log it, never speak it to the user


def check_user_at_satellite(user_id: str, satellite_ip: str) -> PresenceCheck:
    """
    Cross-check a claimed speaker identity (from STT speaker-ID) against
    independently-derived presence data, at request time.

    This does NOT gate whether a request executes — a handler should still
    do the thing it was asked to do regardless of the result. It exists so
    a handler can decide whether user_id is trustworthy enough to bind
    identity-sensitive behaviour to: if speaker-ID says "Alice" but
    Alice's tag currently resolves to a different room, something's wrong
    (misidentification, a stale tag, a guest triggering a false match) and
    the safe thing is to silently fall back to ordinary, non-personalised
    behaviour rather than propagate the mismatch across the house (e.g. by
    starting a follow-me session that then "follows" the wrong person).
    """
    if user_id == "unknown":
        return PresenceCheck(False, None, "speaker unidentified")

    location = get_user_location(user_id)
    if location is None:
        return PresenceCheck(False, None, "no presence data for user")

    if location == satellite_ip:
        return PresenceCheck(True, location, "location matches request origin")

    return PresenceCheck(
        False, location,
        f"speaker-ID says {user_id!r} but presence resolves to {location}, not {satellite_ip}",
    )


# ---------------------------------------------------------------------------
# Media sessions — "follow me" playback bookkeeping
# ---------------------------------------------------------------------------
# Core because more than one thing needs to read/write it outside of a
# single request: orchestration.py's media_follow_poller moves sessions as
# presence changes, and any playback skill can start/end one. Only one
# playback backend (skills/music.py) exists today, so the actual MPD
# move/fade mechanics live there — this section only owns the bookkeeping
# of whose session is where, the same split triggers draws between
# "when/who/where" (core) and "what it means" (skill).

@dataclass(frozen=True)
class MediaSession:
    user_id: str
    content_ref: dict
    current_satellite: str
    started_at: datetime
    follow_enabled: bool


def _media_session_from_row(row: sqlite3.Row) -> MediaSession:
    try:
        content_ref = json.loads(row["content_ref"])
    except json.JSONDecodeError:
        log.error(
            "media_sessions row for user=%r has unparseable content_ref — treating as {}",
            row["user_id"],
        )
        content_ref = {}
    return MediaSession(
        user_id=row["user_id"],
        content_ref=content_ref,
        current_satellite=row["current_satellite"],
        started_at=datetime.fromisoformat(row["started_at"]),
        follow_enabled=bool(row["follow_enabled"]),
    )


def start_media_session(
    user_id: str, content_ref: dict, satellite_ip: str, follow_enabled: bool,
) -> None:
    """
    Create or replace the follow-tracked session for a user.

    Only call this for a resolved identity worth tracking — skip entirely
    for user_id == "unknown" (nothing to follow), or when
    check_user_at_satellite() didn't trust the claimed identity. See
    skills/music.py's handle_play_music for the calling convention this
    is designed around.
    """
    conn = _db()
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO media_sessions
            (user_id, content_ref, current_satellite, started_at, follow_enabled)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (user_id) DO UPDATE SET
            content_ref = excluded.content_ref,
            current_satellite = excluded.current_satellite,
            started_at = excluded.started_at,
            follow_enabled = excluded.follow_enabled
        """,
        (user_id, json.dumps(content_ref), satellite_ip, now, int(follow_enabled)),
    )
    conn.commit()
    log.debug(
        "Media session started: user=%r at=%s follow=%s", user_id, satellite_ip, follow_enabled,
    )


def get_media_session(user_id: str) -> MediaSession | None:
    row = _db().execute(
        "SELECT user_id, content_ref, current_satellite, started_at, follow_enabled "
        "FROM media_sessions WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    return _media_session_from_row(row) if row else None


def get_active_session_at(satellite_ip: str) -> MediaSession | None:
    """
    The follow-enabled session (if any) currently occupying a satellite.

    Used both by skills/music.py (room-conflict check before creating a
    new session) and orchestration.py's media_follow_poller (is the
    destination already occupied by someone else's session?).
    """
    row = _db().execute(
        "SELECT user_id, content_ref, current_satellite, started_at, follow_enabled "
        "FROM media_sessions WHERE current_satellite = ? AND follow_enabled = 1",
        (satellite_ip,),
    ).fetchone()
    return _media_session_from_row(row) if row else None


def get_all_media_sessions() -> list[MediaSession]:
    """Every follow-enabled session, for the media-follow poller to reconcile."""
    rows = _db().execute(
        "SELECT user_id, content_ref, current_satellite, started_at, follow_enabled "
        "FROM media_sessions WHERE follow_enabled = 1"
    ).fetchall()
    return [_media_session_from_row(row) for row in rows]


def move_media_session(user_id: str, new_satellite_ip: str) -> None:
    """
    Update bookkeeping after playback has already been physically moved to
    new_satellite_ip (see skills/music.py's move_session_playback). This
    function only ever updates the row — it does not touch MPD itself.
    """
    conn = _db()
    conn.execute(
        "UPDATE media_sessions SET current_satellite = ? WHERE user_id = ?",
        (new_satellite_ip, user_id),
    )
    conn.commit()
    log.debug("Media session moved: user=%r -> %s", user_id, new_satellite_ip)


def end_media_session(user_id: str) -> None:
    """Drop a user's session — playback stopped, presence lost, or suppressed by a conflict."""
    conn = _db()
    conn.execute("DELETE FROM media_sessions WHERE user_id = ?", (user_id,))
    conn.commit()
    log.debug("Media session ended: user=%r", user_id)

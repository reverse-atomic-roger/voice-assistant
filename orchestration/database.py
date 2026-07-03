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
skill's trigger means. Two things about a trigger are core concerns and get
real columns:
  - `fires_at`     — when, because the poller has to index/query on it.
  - `satellite_ip` — where any resulting announcement gets delivered, because
    routing audio to a satellite is core infrastructure (orchestration.py /
    audio_io.py), not skill semantics.
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
from datetime import datetime, timezone
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
    id           INTEGER PRIMARY KEY,
    skill        TEXT    NOT NULL,               -- opaque skill identifier, matches TriggerHandler.skill_name
    trigger_key  TEXT    NOT NULL,                -- skill-local key, not interpreted by the core (lets a skill find/cancel its own trigger later)
    fires_at     TEXT    NOT NULL,                -- UTC ISO 8601
    satellite_ip TEXT    NOT NULL,                -- where to deliver the resulting announcement
    payload      TEXT    NOT NULL DEFAULT '{}',   -- opaque JSON owned by the skill
    fired        INTEGER NOT NULL DEFAULT 0
);

-- Speeds up the poller's "find due, unfired triggers" query.
CREATE INDEX IF NOT EXISTS idx_triggers_fires_at ON triggers (fires_at)
    WHERE fired = 0;
"""


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
    One due (or pending) row from the triggers table, with `payload` already
    decoded from JSON so callers never touch the storage format directly.
    """
    id: int
    skill: str
    trigger_key: str
    fires_at: datetime
    satellite_ip: str
    payload: dict


def add_trigger(
    skill: str,
    trigger_key: str,
    fires_at: datetime,
    satellite_ip: str,
    payload: dict | None = None,
) -> int:
    """
    Schedule a future event and return its row id.

    skill        — identifier matching a registered TriggerHandler.skill_name
                   (see skills/registry.py). The poller uses this to look up
                   which handler to call when the trigger fires.
    trigger_key  — skill-local identifier (e.g. a label or generated id).
                   Never interpreted by the core; it exists so a skill can
                   later find or cancel its own trigger without needing a
                   bespoke lookup function added here.
    fires_at     — must be a UTC-aware datetime; stored as an ISO 8601 string.
    satellite_ip — where the resulting announcement should be delivered when
                   the trigger fires. This is a real column (not part of
                   payload) because routing audio to a satellite is core
                   delivery infrastructure, not skill semantics.
    payload      — arbitrary JSON-serialisable dict owned entirely by the
                   skill. Stored and returned verbatim; the core never reads
                   or interprets its contents.
    """
    fires_at_str = fires_at.astimezone(timezone.utc).isoformat()
    payload_str = json.dumps(payload or {})
    cur = _db().execute(
        "INSERT INTO triggers (skill, trigger_key, fires_at, satellite_ip, payload, fired) "
        "VALUES (?, ?, ?, ?, ?, 0)",
        (skill, trigger_key, fires_at_str, satellite_ip, payload_str),
    )
    _db().commit()
    log.debug(
        "Trigger added: id=%d skill=%r trigger_key=%r fires_at=%s satellite=%s",
        cur.lastrowid, skill, trigger_key, fires_at_str, satellite_ip,
    )
    return cur.lastrowid


def get_due_triggers(now: datetime) -> list[Trigger]:
    """
    Return all unfired triggers whose fires_at is at or before `now`, as
    Trigger objects with `payload` already decoded from JSON.

    `now` should be UTC-aware; compared as ISO 8601 strings (sorts correctly).
    """
    now_str = now.astimezone(timezone.utc).isoformat()
    rows = _db().execute(
        "SELECT id, skill, trigger_key, fires_at, satellite_ip, payload FROM triggers "
        "WHERE fired = 0 AND fires_at <= ?",
        (now_str,),
    ).fetchall()

    due: list[Trigger] = []
    for row in rows:
        try:
            payload = json.loads(row["payload"])
        except json.JSONDecodeError:
            log.error(
                "Trigger id=%d (skill=%r) has unparseable payload — treating as {}",
                row["id"], row["skill"],
            )
            payload = {}
        due.append(Trigger(
            id=row["id"],
            skill=row["skill"],
            trigger_key=row["trigger_key"],
            fires_at=datetime.fromisoformat(row["fires_at"]),
            satellite_ip=row["satellite_ip"],
            payload=payload,
        ))
    return due


def mark_trigger_fired(trigger_id: int) -> None:
    """Mark a trigger as fired so the poller doesn't re-announce it."""
    _db().execute("UPDATE triggers SET fired = 1 WHERE id = ?", (trigger_id,))
    _db().commit()
    log.debug("Trigger %d marked fired", trigger_id)

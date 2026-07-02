#!/usr/bin/env python3
"""
database.py

Owns the SQLite connection for the voice assistant. Manages connection
lifecycle plus the schema and queries that are genuinely orchestrator-core.
Right now that's just timers, because orchestration.py's timer poller
queries them directly, outside of any skill handler.

Skill-owned persistence does NOT live here. A skill that needs its own
table calls `database.register_schema(...)` once at module import time with
its own CREATE TABLE DDL, and does its own reads/writes through
`database.get_connection()`. That keeps this file from growing a new
bespoke function every time someone adds a skill with storage needs — the
same reason skill handlers live in skills/*.py instead of orchestration.py.
See skills/README.md, "Owning your own persistence", for the pattern and
skills/lists.py for a worked example.

Connection is opened once at startup via init() and reused for the lifetime
of the process. SQLite in WAL mode handles the concurrent reads the timer
polling loop and inbound requests both need without contention.

Thread safety: asyncio runs everything on one thread, so no locking is
needed here beyond what SQLite itself provides.

Schema notes:
  - timers.fired is an INTEGER (0/1) rather than deleting rows on fire,
    so there is a record of past timers for any future diagnostics.
  - All timestamps are UTC ISO 8601 strings — SQLite has no native datetime
    type, and strings sort correctly for the range queries the timer poller
    needs.
"""

import logging
import sqlite3
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

CREATE TABLE IF NOT EXISTS timers (
    id               INTEGER PRIMARY KEY,
    label            TEXT    NOT NULL,
    fires_at         TEXT    NOT NULL,   -- UTC ISO 8601
    satellite_id     TEXT    NOT NULL,   -- IP of originating satellite
    fired            INTEGER NOT NULL DEFAULT 0,
    duration_label   TEXT    NOT NULL    -- Human-readable duration, e.g. "3 minutes"
);

-- Speeds up the poller's "find due, unfired timers" query.
CREATE INDEX IF NOT EXISTS idx_timers_fires_at ON timers (fires_at)
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
    the orchestrator itself (like timers) still belongs as a real function
    in this file.

    Row factory is sqlite3.Row (dict-like access by column name). Commits
    are the caller's responsibility, same as every function in this file.
    """
    return _db()


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------
# Kept here rather than moved into skills/timer.py: orchestration.py's
# timer_poller queries these directly, outside of any skill handler, which
# makes this genuinely core orchestrator state rather than skill-owned
# persistence.

def add_timer(label: str, fires_at: datetime, satellite_id: str, duration_label: str = "") -> int:
    """
    Insert a new timer and return its row id.

    fires_at must be a UTC-aware datetime; it is stored as an ISO 8601 string.
    duration_label is a pre-formatted spoken string (e.g. "3 minutes") stored so
    the poller can build a meaningful announcement for unlabelled timers without
    recomputing it on every poll cycle.
    """
    fires_at_str = fires_at.astimezone(timezone.utc).isoformat()
    cur = _db().execute(
        "INSERT INTO timers (label, fires_at, satellite_id, fired, duration_label) VALUES (?, ?, ?, 0, ?)",
        (label, fires_at_str, satellite_id, duration_label),
    )
    _db().commit()
    log.debug("Timer added: id=%d label=%r fires_at=%s satellite=%s duration_label=%r",
              cur.lastrowid, label, fires_at_str, satellite_id, duration_label)
    return cur.lastrowid


def get_due_timers(now: datetime) -> list[sqlite3.Row]:
    """
    Return all unfired timers whose fires_at is at or before `now`.

    Each row has: id, label, fires_at, satellite_id, duration_label.
    `now` should be UTC-aware; compared as ISO 8601 strings (sorts correctly).
    """
    now_str = now.astimezone(timezone.utc).isoformat()
    return _db().execute(
        "SELECT id, label, fires_at, satellite_id, duration_label FROM timers "
        "WHERE fired = 0 AND fires_at <= ?",
        (now_str,),
    ).fetchall()


def mark_timer_fired(timer_id: int) -> None:
    """Mark a timer as fired so the poller doesn't re-announce it."""
    _db().execute("UPDATE timers SET fired = 1 WHERE id = ?", (timer_id,))
    _db().commit()
    log.debug("Timer %d marked fired", timer_id)

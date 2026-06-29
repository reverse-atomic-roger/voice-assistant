#!/usr/bin/env python3
"""
database.py

Owns the SQLite connection for the voice assistant. Manages schema
initialisation and exposes query functions for timers and lists.

All SQL is in this module. Handlers call these functions and work entirely
with plain Python values — no SQL leaks out.

Connection is opened once at startup via init() and reused for the lifetime
of the process. SQLite in WAL mode handles the concurrent reads the timer
polling loop and inbound requests both need without contention.

Thread safety: asyncio runs everything on one thread, so no locking is
needed here beyond what SQLite itself provides.

Schema notes:
  - timers.fired is an INTEGER (0/1) rather than deleting rows on fire,
    so there is a record of past timers for any future diagnostics.
  - lists rows are created implicitly when an item is added to a named list
    that doesn't yet exist (upsert on name).
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


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
-- App data tables only. Media tables (tracks, track_features, etc.) are
-- created by the media indexer in Phase 2 and are not initialised here.

CREATE TABLE IF NOT EXISTS lists (
    id         INTEGER PRIMARY KEY,
    name       TEXT    UNIQUE NOT NULL COLLATE NOCASE,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS list_items (
    id         INTEGER PRIMARY KEY,
    list_id    INTEGER NOT NULL REFERENCES lists(id),
    content    TEXT    NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);

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


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def init() -> None:
    """
    Open the database, enable WAL mode, and run the schema DDL.

    Creates the data directory and database file if they don't exist.
    Raises on any failure — intended to be called once at orchestrator
    startup so a bad DB path fails loudly before any requests are served.
    """
    global _conn

    if _conn is not None:
        raise RuntimeError("database.init() called more than once")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    _conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    _conn.row_factory = sqlite3.Row

    # WAL mode: readers don't block writers and vice versa. Safe for a
    # single-process asyncio service; good habit for when the media indexer
    # runs as a separate process in Phase 2.
    _conn.execute("PRAGMA journal_mode=WAL")

    # Enforce foreign key constraints — SQLite disables them by default.
    _conn.execute("PRAGMA foreign_keys=ON")

    _conn.executescript(_SCHEMA)
    _conn.commit()

    log.info("Database initialised at %s", DB_PATH)


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


# ---------------------------------------------------------------------------
# Timers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Lists
# ---------------------------------------------------------------------------

def _get_or_create_list(name: str) -> int:
    """
    Return the id of the list with the given name, creating it if needed.
    Name comparison is case-insensitive (COLLATE NOCASE on the column).
    """
    now = datetime.now(timezone.utc).isoformat()
    # INSERT OR IGNORE leaves the row untouched if the name already exists.
    _db().execute(
        "INSERT OR IGNORE INTO lists (name, created_at) VALUES (?, ?)",
        (name, now),
    )
    _db().commit()
    row = _db().execute(
        "SELECT id FROM lists WHERE name = ?", (name,)
    ).fetchone()
    return row["id"]


def add_list_item(list_name: str, content: str) -> None:
    """
    Add an item to the named list, creating the list if it doesn't exist.
    """
    list_id = _get_or_create_list(list_name)
    now = datetime.now(timezone.utc).isoformat()
    _db().execute(
        "INSERT INTO list_items (list_id, content, done, created_at) VALUES (?, ?, 0, ?)",
        (list_id, content, now),
    )
    _db().commit()
    log.debug("List item added: list=%r item=%r", list_name, content)


def get_list_items(list_name: str) -> list[str]:
    """
    Return the content of all undone items in the named list, in insertion
    order. Returns an empty list if the list doesn't exist or has no items.
    """
    row = _db().execute(
        "SELECT id FROM lists WHERE name = ?", (list_name,)
    ).fetchone()

    if row is None:
        return []

    rows = _db().execute(
        "SELECT content FROM list_items WHERE list_id = ? AND done = 0 "
        "ORDER BY id ASC",
        (row["id"],),
    ).fetchall()

    return [r["content"] for r in rows]

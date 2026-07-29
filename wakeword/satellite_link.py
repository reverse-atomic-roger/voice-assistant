#!/usr/bin/env python3
"""
satellite_link.py

Persistent, satellite-initiated connection to orchestration.py's satellite
link listener (see orchestration.py's SATELLITE_LINK_HOST/PORT and
handle_satellite_link_connection()).

Deliberately generic rather than presence-specific: one newline-delimited
JSON message per line, dispatched on the orchestration side by a "type"
field. presence_scanner.py is the first user of this (message type
"presence"), but the intent is for this to be the one satellite -> 
orchestration channel, so a later feature (e.g. a datapad-style on-device
screen reporting its own status) can share this connection instead of
inventing a second protocol from scratch.

Reconnects with exponential backoff if the connection drops or
orchestration is unreachable at startup — expected during orchestration
restarts, and should never crash the satellite process, which still needs
to serve wake word / audio duties regardless of whether this link is
currently up.

Delivery is best-effort, not guaranteed: send() silently drops a message
if there's no current connection, rather than queuing it for later. This
is the right tradeoff for presence data (a dropped reading is superseded
by the next one seconds later) but would NOT be the right tradeoff for a
future message type needing at-least-once delivery (e.g. a datapad
action) — that would need its own queue/ack layer built on top of this,
not an assumption that this transport already provides one.

This module does not start itself — it is imported by presence_scanner.py
(and any future sender) and run for the lifetime of the process by
satellite_main.py.

Dependencies: none beyond the standard library.
"""

import asyncio
import json
import logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: orchestration's satellite link host/port. Must match
# SATELLITE_LINK_HOST/SATELLITE_LINK_PORT in orchestration.py.
ORCHESTRATION_HOST = "127.0.0.1"
ORCHESTRATION_PORT = 10303

# CONFIGURE: reconnect backoff, in seconds. Starts at the first value and
# doubles after each consecutive failed attempt, up to the cap.
RECONNECT_INITIAL_DELAY_S = 1.0
RECONNECT_MAX_DELAY_S = 30.0

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

_writer: asyncio.StreamWriter | None = None
_writer_lock: asyncio.Lock = asyncio.Lock()


async def send(message: dict) -> bool:
    """
    Send one message (as a JSON line) over the current connection.

    Returns True if it was actually written, False if there's currently no
    connection (orchestration unreachable, mid-reconnect, etc.). Callers
    that only care about "best effort, most recent value wins"
    (presence_scanner.py today) can ignore the return value; a future
    caller needing to know a message definitely went out should check it
    and decide its own retry policy — this function does not retry.
    """
    async with _writer_lock:
        if _writer is None:
            return False
        try:
            _writer.write(json.dumps(message).encode("utf-8") + b"\n")
            await _writer.drain()
            return True
        except (ConnectionError, OSError) as exc:
            log.warning("satellite_link: send failed: %s", exc)
            return False


async def run() -> None:
    """
    Maintain a connection to orchestration for the lifetime of the process,
    reconnecting with exponential backoff on any drop or failure to
    connect.

    Never raises — see this module's docstring for why a satellite with no
    working link to orchestration should still serve wake word / audio
    duties normally rather than crash.
    """
    global _writer

    delay = RECONNECT_INITIAL_DELAY_S
    while True:
        try:
            reader, writer = await asyncio.open_connection(
                ORCHESTRATION_HOST, ORCHESTRATION_PORT,
            )
            async with _writer_lock:
                _writer = writer
            log.info(
                "satellite_link: connected to orchestration at %s:%d",
                ORCHESTRATION_HOST, ORCHESTRATION_PORT,
            )
            delay = RECONNECT_INITIAL_DELAY_S  # reset backoff after a clean connect

            # This link is satellite -> orchestration only today (no replies
            # expected back), so just wait for the connection to drop
            # rather than reading anything meaningful. read() with no
            # argument blocks until EOF (peer closed) or raises (dead
            # connection) — either way, that's our cue to reconnect.
            await reader.read()

        except (ConnectionError, OSError) as exc:
            log.warning("satellite_link: connection error: %s", exc)
        finally:
            async with _writer_lock:
                if _writer is not None:
                    _writer.close()
                    try:
                        await _writer.wait_closed()
                    except Exception:
                        pass
                _writer = None

        log.info("satellite_link: reconnecting in %.1fs", delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, RECONNECT_MAX_DELAY_S)

#!/usr/bin/env python3
"""
mpd_control.py

Ducks mpd's volume while the satellite needs the speaker to itself — during
a wake-word capture window, or while playing an audio clip pushed from the
server (timer done, command acknowledged, etc.) — and restores it
afterwards.

Uses python-mpd2, a synchronous client, via asyncio.to_thread(). mpd's
control protocol is a trivial, low-frequency exchange (a status query and a
setvol), so a blocking client run off the event loop is simpler than an
async one and has no real downside here.

Concurrency model:
    duck() / unduck() are reference-counted rather than a simple flag,
    because wakeword_stream and audio_receiver run concurrently
    (satellite_main's TaskGroup) and can legitimately want to duck at the
    same time — e.g. a "timer done" clip lands while the user is mid
    command. The first duck() call reads and stores mpd's current volume;
    later, nested duck() calls just bump the count. Volume is only
    restored once every matching unduck() has been called, and only
    if something was actually lowered.

Usage:
    async with mpd_control.ducking():
        ...do the thing that needs the speaker...

Dependencies:
    pip install python-mpd2
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from mpd import MPDClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: mpd connection details. mpd runs on the satellite itself.
MPD_HOST = "localhost"
MPD_PORT = 6600

# CONFIGURE: volume (0-100) to duck down to. Only ducks if mpd's current
# volume is currently above this.
DUCK_VOLUME = 10

# ---------------------------------------------------------------------------

_duck_lock = asyncio.Lock()
_duck_count = 0
_pre_duck_volume: int | None = None


def _connect() -> MPDClient:
    client = MPDClient()
    client.timeout = 2
    client.connect(MPD_HOST, MPD_PORT)
    return client


def _get_volume() -> int:
    """
    Blocking: query mpd's current volume (0-100, or -1 if truly unknown).

    Tries the dedicated `getvol` command first — the current, reliable way
    to read volume. `status`'s `volume` field can read back -1 (or be
    absent) on mpd setups without a single global mixer to report through
    the status snapshot, even while the actual output volume is fine, so
    relying on it alone silently made duck() think there was nothing to
    duck. Falls back to `status` for older mpd that predates `getvol`.
    """
    client = _connect()
    try:
        vol = -1
        try:
            result = client.getvol()
            vol = int(result.get("volume", -1))
        except Exception as exc:
            log.debug("getvol unavailable, falling back to status: %s", exc)

        if vol < 0:
            status = client.status()
            vol = int(status.get("volume", -1))

        return vol
    finally:
        try:
            client.close()
            client.disconnect()
        except Exception:
            pass


def _set_volume(vol: int) -> None:
    """Blocking: set mpd's volume (0-100)."""
    client = _connect()
    try:
        client.setvol(vol)
    finally:
        try:
            client.close()
            client.disconnect()
        except Exception:
            pass


async def duck() -> None:
    """
    Lower mpd's volume to DUCK_VOLUME if it's currently louder than that,
    remembering the prior volume for restore. Safe to call from multiple
    overlapping contexts — only the first caller (per ducked period)
    captures the volume to restore; later callers just bump the ref count.

    Ducking is a nice-to-have layered on top of the wake-word/playback
    path, not a prerequisite for it — so ANY failure talking to mpd
    (unreachable, requires a password, protocol hiccup, whatever) is
    swallowed here rather than propagated. python-mpd2's exceptions don't
    all inherit from OSError (auth/protocol errors are mpd.base.MPDError,
    a separate hierarchy), so this catches Exception broadly on purpose —
    a narrower list already let one of these through once and it took the
    whole satellite process down via the TaskGroup in satellite_main.py.
    """
    global _duck_count, _pre_duck_volume

    async with _duck_lock:
        if _duck_count == 0:
            try:
                current = await asyncio.to_thread(_get_volume)
                if current > DUCK_VOLUME:
                    await asyncio.to_thread(_set_volume, DUCK_VOLUME)
                    _pre_duck_volume = current
                    log.info("Ducked mpd volume %d -> %d", current, DUCK_VOLUME)
                else:
                    _pre_duck_volume = None
                    log.debug("mpd volume already <= %d — nothing to duck", DUCK_VOLUME)
            except Exception as exc:
                log.warning("mpd ducking failed, continuing without it: %s", exc)
                _pre_duck_volume = None

        _duck_count += 1


async def unduck() -> None:
    """
    Undo one duck() call. Once every matching duck() call has been undone,
    restores whatever volume was in effect before ducking began (if any
    was actually lowered). Never raises — see duck() for why.
    """
    global _duck_count, _pre_duck_volume

    async with _duck_lock:
        if _duck_count == 0:
            log.warning("unduck() called with no matching duck() — ignoring")
            return

        _duck_count -= 1
        if _duck_count == 0 and _pre_duck_volume is not None:
            try:
                await asyncio.to_thread(_set_volume, _pre_duck_volume)
                log.info("Restored mpd volume to %d", _pre_duck_volume)
            except Exception as exc:
                log.warning(
                    "Failed to restore mpd volume — it will stay at %d until "
                    "the next successful duck/unduck cycle: %s",
                    DUCK_VOLUME, exc,
                )
            finally:
                _pre_duck_volume = None


@asynccontextmanager
async def ducking():
    """
    Async context manager wrapping duck()/unduck() so call sites can't
    forget to restore volume, even on an exception.

        async with mpd_control.ducking():
            ...
    """
    await duck()
    try:
        yield
    finally:
        await unduck()

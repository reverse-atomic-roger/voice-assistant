#!/usr/bin/env python3
"""
presence_scanner.py

Scans for known BLE presence tags (fixed-MAC advertisers — see KNOWN_TAGS
below) and reports sightings to orchestration over satellite_link.py, for
database.py's presence tables and everything built on top of them
(skills/locate.py, skills/music.py's follow-me sessions). See
ARCHITECTURE.md for the full presence design.

Uses bleak's continuous scanner with a detection callback rather than
polling bleak.discover() in a loop — the callback fires as advertisements
arrive with no busy-loop, and lets scanning run continuously in the
background while a separate periodic task (report_loop) decides how often
to actually forward readings. This decouples "how fast can BLE tell us
something" from "how often does the rest of the system need to know",
matching the same reasoning behind report cadence vs. resolution
hysteresis in database.py.

Assumes tags advertise with a fixed (not randomised/rotating) MAC address —
true of the simple dedicated BLE tags this was designed for, but notably
NOT true of most phones' own BLE stacks, which use rotating private
addresses for privacy. If phones are ever added as a presence source
alongside dedicated tags, they'll need a different mechanism entirely (a
companion app reporting over satellite_link directly, most likely) rather
than being matched here by MAC.

Dependencies:
    pip install bleak
"""

import asyncio
import logging
import time

from bleak import BleakScanner
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData

import satellite_link

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: known presence tags, MAC address (as bleak reports it, e.g.
# "AA:BB:CC:DD:EE:FF" — uppercase, colon-separated) -> user_id. user_id
# here must match whatever the STT server's speaker-ID step produces for
# the same person (see stt/stt_server.py / enroll_speaker.py), since
# database.check_user_at_satellite() compares the two directly — a
# mismatched user_id here will silently mean "follow me" never trusts this
# person's presence, without any obvious error to point at, so double-check
# this against the actual enrolled speaker-ID strings.
KNOWN_TAGS: dict[str, str] = {
    # "AA:BB:CC:DD:EE:FF": "alice",
    # "11:22:33:44:55:66": "bob",
}

# CONFIGURE: how often sightings are forwarded to orchestration. Scanning
# itself is continuous (see module docstring); this only controls
# reporting cadence. Should be comfortably under database.py's
# STALE_SECONDS (45s at time of writing) so a resolved location doesn't go
# stale between reports under normal operation.
REPORT_INTERVAL_S = 5.0

# CONFIGURE: a sighting is not reported if it's older than this, even if
# it's the most recent one seen for a tag — an advertisement seen a while
# ago and never refreshed since likely means the tag walked out of range
# between report cycles, and reporting a stale RSSI would be actively
# misleading rather than just uninformative.
SIGHTING_MAX_AGE_S = REPORT_INTERVAL_S * 2

# CONFIGURE: reconnect-style backoff for the BLE scanner itself (BlueZ/
# D-Bus unavailable, adapter missing, permissions issue, etc.), separate
# from satellite_link.py's own backoff for the network side.
SCAN_RETRY_INITIAL_DELAY_S = 1.0
SCAN_RETRY_MAX_DELAY_S = 30.0

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# mac -> (rssi, monotonic timestamp of last sighting). Written by the
# detection callback, read (and implicitly aged out) by report_loop.
_latest: dict[str, tuple[int, float]] = {}


def _on_detection(device: BLEDevice, advertisement: AdvertisementData) -> None:
    """
    bleak detection callback — fires for every BLE advertisement bleak's
    backend sees, from every device in range, not just known tags. Filtered
    to KNOWN_TAGS here (rather than in report_loop) so _latest never grows
    with irrelevant neighbours' phones/earbuds/etc.
    """
    mac = device.address.upper()
    if mac not in KNOWN_TAGS:
        return
    _latest[mac] = (advertisement.rssi, time.monotonic())


async def report_loop() -> None:
    """
    Wake every REPORT_INTERVAL_S and forward the latest still-fresh
    reading for each known tag to orchestration via satellite_link.send().

    Runs until cancelled or an unexpected exception propagates (caught by
    run(), see below). A send() failure is silently dropped — see
    satellite_link.py's docstring on best-effort delivery — not retried
    here, since the next cycle's reading supersedes it anyway.
    """
    while True:
        await asyncio.sleep(REPORT_INTERVAL_S)
        now = time.monotonic()

        for mac, user_id in KNOWN_TAGS.items():
            sighting = _latest.get(mac)
            if sighting is None:
                continue
            rssi, seen_at = sighting
            if now - seen_at > SIGHTING_MAX_AGE_S:
                continue  # stale — tag likely out of range since last report

            await satellite_link.send({
                "type": "presence",
                "user_id": user_id,
                "rssi": rssi,
            })


async def run() -> None:
    """
    Run the BLE scanner and the periodic report loop for the lifetime of
    the process, retrying with backoff on failure.

    Deliberately never lets a scanning failure propagate out of this
    function — unlike wakeword_stream.run()/audio_receiver.run(), whose
    failure genuinely means "this satellite is no longer functional",
    presence scanning is a nice-to-have: a satellite with no Bluetooth
    adapter, or a transient BlueZ/D-Bus/permissions problem, should still
    handle wake word and audio normally. See satellite_main.py's
    docstring — this task is added to the same TaskGroup as the core two,
    but is designed to never raise into it.
    """
    delay = SCAN_RETRY_INITIAL_DELAY_S

    while True:
        try:
            async with BleakScanner(detection_callback=_on_detection):
                log.info(
                    "BLE presence scanner started (%d known tag(s))", len(KNOWN_TAGS),
                )
                delay = SCAN_RETRY_INITIAL_DELAY_S  # reset backoff after a clean start
                await report_loop()  # runs forever unless it raises
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.error("BLE scanner failed, retrying in %.1fs: %s", delay, exc)
            await asyncio.sleep(delay)
            delay = min(delay * 2, SCAN_RETRY_MAX_DELAY_S)

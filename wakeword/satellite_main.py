#!/usr/bin/env python3
"""
satellite_main.py

Entrypoint for the satellite device. Initialises the shared audio_io module,
then runs four things concurrently for the lifetime of the process:

    - wakeword_stream.run()    — listens for the wake word, streams commands
                                  to the STT server, plays capture earcons
    - audio_receiver.run()     — Wyoming server; plays canned responses now,
                                  TTS audio later, as sent by orchestration
    - satellite_link.run()     — persistent connection to orchestration for
                                  satellite-initiated messages (presence
                                  reports today; a natural home for other
                                  satellite-originated messages later)
    - presence_scanner.run()   — scans for known BLE presence tags and
                                  reports sightings over satellite_link

Both audio-related tasks consume the same audio_io module for
capture/playback, so there is exactly one PyAudio instance and one
serialised output path in this process.

This is the boundary where startup failures and shutdown are handled — the
first two run() functions raise freely; this module is what catches and
logs. satellite_link.run() and presence_scanner.run() are different: they
are designed to never raise (see their own docstrings) — a satellite with
no working Bluetooth adapter, or no network path to orchestration, should
still handle wake word and audio normally, so their failures are logged
and retried internally rather than propagating into this TaskGroup. They
are added to the same TaskGroup as the other two purely so the whole
process still shuts down together on Ctrl-C / SIGTERM, not because a
failure in one is meant to bring down the others.

Dependencies:
    pip install openwakeword pyaudio wyoming py-silero-vad-lite soundfile numpy bleak
"""

import asyncio
import logging
import sys

import audio_io
import audio_receiver
import presence_scanner
import satellite_link
import wakeword_stream

log = logging.getLogger(__name__)


async def main() -> None:
    audio_io.init()
    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(wakeword_stream.run(), name="wakeword")
            tg.create_task(audio_receiver.run(), name="audio_receiver")
            tg.create_task(satellite_link.run(), name="satellite_link")
            tg.create_task(presence_scanner.run(), name="presence_scanner")
    finally:
        audio_io.shutdown()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")
    except ExceptionGroup as eg:
        for exc in eg.exceptions:
            log.error("Fatal error", exc_info=exc)
        sys.exit(1)
    except Exception:
        log.exception("Fatal error")
        sys.exit(1)

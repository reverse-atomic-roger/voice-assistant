#!/usr/bin/env python3
"""
satellite_main.py

Entrypoint for the satellite device. Initialises the shared audio_io module,
then runs two things concurrently for the lifetime of the process:

    - wakeword_stream.run()   — listens for the wake word, streams commands
                                 to the STT server, plays capture earcons
    - audio_receiver.run()    — Wyoming server; plays canned responses now,
                                 TTS audio later, as sent by orchestration

Both consume the same audio_io module for capture/playback, so there is
exactly one PyAudio instance and one serialised output path in this process.

This is the boundary where startup failures and shutdown are handled — the
two run() functions raise freely; this module is what catches and logs.

Dependencies:
    pip install openwakeword pyaudio wyoming py-silero-vad-lite soundfile numpy
"""

import asyncio
import logging
import sys

import audio_io
import audio_receiver
import wakeword_stream

log = logging.getLogger(__name__)


async def main() -> None:
    audio_io.init()
    try:
        # If either task raises, cancel the other and propagate — a dead
        # wakeword loop or a dead audio receiver both mean the satellite is
        # no longer functional, so there's no good reason to keep one alive
        # without the other.
        async with asyncio.TaskGroup() as tg:
            tg.create_task(wakeword_stream.run(), name="wakeword")
            tg.create_task(audio_receiver.run(), name="audio_receiver")
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

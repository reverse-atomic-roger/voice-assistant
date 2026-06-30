#!/usr/bin/env python3
"""
audio_io.py

Owns the satellite's single PyAudio instance, input stream, and output
playback path. Both wakeword_stream.py (capture) and audio_receiver.py
(playback of canned/TTS audio) use this module rather than opening their
own PyAudio() instances.

Rationale for centralising here:
  - Only one PyAudio() instance should exist per process; multiple
    instances fighting over the same ALSA device is a known source of
    flaky behaviour on the Pi.
  - Playback needs to be serialised. If a canned response from the
    orchestrator arrives while an earcon is mid-playback, two concurrent
    writes to the output stream will garble audio or raise an ALSA error.
    play_pcm() takes an asyncio.Lock so callers queue rather than collide.
  - Capture (input) and playback (output) are separate streams/directions
    and do not block each other.

Capture architecture:
  A dedicated thread (_run_capture_thread) reads from PyAudio in a tight
  loop and pushes frames onto an asyncio.Queue. wakeword_stream.py pulls
  from this queue with `await audio_io.capture_queue().get()` rather than
  calling asyncio.to_thread() on every frame. This avoids ~750 thread-pool
  submissions per minute and means a stalled ALSA/USB read in the capture
  thread never blocks the event loop.

  init() must be called from within a running asyncio context (i.e. from
  inside a coroutine) because it calls asyncio.get_running_loop() to bind
  the capture thread to the correct loop.

This module does not start any loops itself — it is imported by
wakeword_stream.py and audio_receiver.py, and initialised once by
satellite_main.py at startup.

The listen_after asyncio.Event is the coordination point for clarification
prompts. audio_receiver sets it when the orchestrator signals that a reply
is expected; wakeword_stream clears it and skips the wake word gate when it
fires. This lets a single wake word activation carry a full multi-turn
exchange without the user needing to re-trigger.

Dependencies:
    pip install pyaudio numpy
"""

import asyncio
import logging
import threading

import pyaudio

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: PyAudio input device index (run `python -m sounddevice` to list)
MIC_DEVICE_INDEX = 1

# CONFIGURE: PyAudio output device index. None = system default output.
OUTPUT_DEVICE_INDEX = None

# Fixed — required by openwakeword and Silero VAD; do not change without
# also changing the wakeword model and VAD configuration.
INPUT_SAMPLE_RATE = 16000
INPUT_CHANNELS = 1
INPUT_SAMPLE_WIDTH = 2  # bytes (int16)

# openwakeword requires exactly 1280-sample input frames (~80 ms)
OWW_FRAME_SAMPLES = 1280

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

# Module-level state, set up by init(). Kept as plain globals rather than a
# class — this module models a single physical device, not a reusable
# object, and every caller in this process wants the same instance.
_pa: pyaudio.PyAudio | None = None
_mic_stream: pyaudio.Stream | None = None
_output_lock: asyncio.Lock | None = None

# Maps (width_bytes, channels) -> a long-lived, kept-open output Stream.
# Reused across calls rather than opened/closed per playback, since
# repeatedly opening PyAudio output streams is the slow part of playback
# and we want the hot path (wake -> earcon) to be snappy.
_output_streams: dict[tuple[int, int], pyaudio.Stream] = {}

_PA_FORMAT_FROM_WIDTH = {
    2: pyaudio.paInt16,
    4: pyaudio.paFloat32,
}

# Queue fed by the dedicated capture thread. wakeword_stream.py awaits .get()
# on this rather than calling to_thread(capture_frame) every 80 ms.
_capture_queue: asyncio.Queue[bytes] | None = None

# Dedicated thread that reads from PyAudio in a tight loop.
_capture_thread: threading.Thread | None = None

# Set to signal the capture thread to exit cleanly on shutdown.
_capture_stop: threading.Event = threading.Event()

# ---------------------------------------------------------------------------
# Clarification listen trigger
# ---------------------------------------------------------------------------
# Set by audio_receiver when the orchestrator sends a listen_after signal.
# Cleared by wakeword_stream when it begins the prompted capture.
# asyncio.Event: safe to set/clear from any coroutine; no thread interaction.
#
# Created at module level so it exists before init() is called — wakeword_stream
# and audio_receiver both reference it at import time.
listen_after: asyncio.Event = asyncio.Event()


def capture_queue() -> "asyncio.Queue[bytes]":
    """
    Return the queue that the capture thread feeds.

    Callers should `await capture_queue().get()` to receive the next
    OWW_FRAME_SAMPLES PCM frame. Never blocks the event loop.
    """
    if _capture_queue is None:
        raise RuntimeError("audio_io.init() must be called before capture_queue()")
    return _capture_queue


def _run_capture_thread(loop: asyncio.AbstractEventLoop) -> None:
    """
    Blocking capture loop that runs on a dedicated thread for the lifetime
    of the process. Reads OWW_FRAME_SAMPLES frames from the mic and pushes
    them onto _capture_queue via the event loop.

    Running on its own thread means a stalled PyAudio read (ALSA underrun,
    USB hiccup, OS focus throttling on Windows) never blocks the event loop
    or starves the asyncio thread pool.
    """
    log.debug("Capture thread started")
    while not _capture_stop.is_set():
        try:
            frame = _mic_stream.read(OWW_FRAME_SAMPLES, exception_on_overflow=False)
        except OSError as exc:
            log.error("Capture thread read error: %s", exc)
            continue
        asyncio.run_coroutine_threadsafe(_capture_queue.put(frame), loop)
    log.debug("Capture thread exiting")


def init() -> None:
    """
    Open the PyAudio instance, the mic input stream, the output lock, and
    the capture thread+queue.

    Must be called once at process startup, from within a running asyncio
    context (i.e. inside a coroutine), before capture_queue() or play_pcm()
    are used. Raises on failure — fail loudly if the audio hardware isn't
    there rather than limping on without it.
    """
    global _pa, _mic_stream, _output_lock, _capture_queue, _capture_thread

    if _pa is not None:
        raise RuntimeError("audio_io.init() called more than once")

    _pa = pyaudio.PyAudio()
    _mic_stream = _pa.open(
        rate=INPUT_SAMPLE_RATE,
        channels=INPUT_CHANNELS,
        format=pyaudio.paInt16,
        input=True,
        input_device_index=MIC_DEVICE_INDEX,
        frames_per_buffer=OWW_FRAME_SAMPLES,
    )
    _output_lock = asyncio.Lock()

    _capture_stop.clear()
    loop = asyncio.get_running_loop()
    _capture_queue = asyncio.Queue(maxsize=50)  # ~4 s of frames; backpressure if consumer stalls
    _capture_thread = threading.Thread(
        target=_run_capture_thread,
        args=(loop,),
        name="mic-capture",
        daemon=True,
    )
    _capture_thread.start()

    log.info(
        "audio_io initialised (input_device=%s, output_device=%s)",
        MIC_DEVICE_INDEX, OUTPUT_DEVICE_INDEX,
    )


def shutdown() -> None:
    """Close all streams, stop the capture thread, and terminate PyAudio.
    Safe to call even if init() was never called."""
    global _pa, _mic_stream

    # Signal and join the capture thread before closing the mic stream it reads.
    _capture_stop.set()
    if _capture_thread is not None and _capture_thread.is_alive():
        _capture_thread.join(timeout=2.0)

    if _mic_stream is not None:
        _mic_stream.stop_stream()
        _mic_stream.close()
        _mic_stream = None

    for stream in _output_streams.values():
        stream.stop_stream()
        stream.close()
    _output_streams.clear()

    if _pa is not None:
        _pa.terminate()
        _pa = None

    log.info("audio_io shut down")


def _get_output_stream(rate: int, width: int, channels: int) -> pyaudio.Stream:
    """
    Return a cached output stream for the given format, opening one if this
    exact format hasn't been used yet. Reused across calls so playback
    latency isn't dominated by stream setup.
    """
    if _pa is None:
        raise RuntimeError("audio_io.init() must be called before playback")

    key = (width, channels)
    stream = _output_streams.get(key)
    if stream is not None:
        return stream

    pa_format = _PA_FORMAT_FROM_WIDTH.get(width)
    if pa_format is None:
        raise ValueError(f"Unsupported sample width for output: {width} bytes")

    stream = _pa.open(
        rate=rate,
        channels=channels,
        format=pa_format,
        output=True,
        output_device_index=OUTPUT_DEVICE_INDEX,
    )
    _output_streams[key] = stream
    log.debug("Opened output stream (rate=%d width=%d channels=%d)", rate, width, channels)
    return stream


async def play_pcm(pcm: bytes, rate: int, width: int, channels: int) -> None:
    """
    Play raw PCM bytes through the satellite's audio output.

    Serialised via _output_lock so concurrent callers (e.g. a wakeword
    earcon and an incoming canned response from the orchestrator) queue
    rather than write to the output stream at the same time.

    width is in bytes: 2 = int16, 4 = float32. Callers must match the
    format their PCM bytes are actually encoded in.
    """
    if _output_lock is None:
        raise RuntimeError("audio_io.init() must be called before play_pcm()")

    async with _output_lock:
        stream = _get_output_stream(rate, width, channels)
        # PyAudio's blocking write() has no asyncio-native equivalent;
        # run it in a thread so we don't block the event loop for the
        # duration of playback.
        await asyncio.to_thread(stream.write, pcm)

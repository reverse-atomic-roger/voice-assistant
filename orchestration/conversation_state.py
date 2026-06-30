#!/usr/bin/env python3
"""
conversation_state.py

Tracks per-satellite clarification state for the voice assistant.

When a handler cannot execute because a required slot is missing (e.g. "set a
timer" with no duration), it raises ClarificationNeeded. The dispatcher stores
that context here, keyed by satellite IP. On the satellite's next utterance,
orchestration checks here before sending the transcript to intent extraction —
if a pending context exists, the transcript is routed to slot-filling instead.

Design constraints:
  - No I/O, no asyncio, no LLM calls — pure in-memory state.
  - Single Python process, single asyncio thread — no locking needed.
  - Expiry is checked lazily on access, not by a background task.
  - Max turns and TTL both enforced. Whichever triggers first wins.

State lifecycle:
    1. Handler raises ClarificationNeeded(intent, slots, question, missing_slot)
    2. dispatcher calls conversation_state.set(satellite_ip, context)
    3. TTS speaks the clarifying question
    4. Next transcript arrives → orchestration calls conversation_state.get(ip)
    5a. Context valid → slot-fill LLM call → handler re-run → clear() on success
    5b. Context expired / max turns hit → clear() → fall through to normal intent extraction
"""

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: seconds of silence/inactivity before a pending clarification expires.
# If the user walks away, we don't want stale context greeting the next person.
CLARIFICATION_TIMEOUT_SECONDS = 30

# CONFIGURE: maximum number of clarifying questions before giving up and resetting.
# Prevents an infinite loop if the model keeps failing to extract the slot.
MAX_CLARIFICATION_TURNS = 2

# ---------------------------------------------------------------------------
# Public exception — raised by handlers, caught by dispatcher
# ---------------------------------------------------------------------------


class ClarificationNeeded(Exception):
    """
    Raised by an intent handler when a required slot is missing and the
    handler wants to ask the user a clarifying question rather than failing.

    Attributes:
        intent      — the original intent name (e.g. "timer")
        slots       — slots extracted so far (may be partial)
        missing_slot — the specific slot key that is absent (e.g. "duration_seconds")
        question    — the TTS string to speak to the user
    """

    def __init__(
        self,
        intent: str,
        slots: dict,
        missing_slot: str,
        question: str,
    ) -> None:
        super().__init__(question)
        self.intent = intent
        self.slots = slots
        self.missing_slot = missing_slot
        self.question = question


# ---------------------------------------------------------------------------
# State container
# ---------------------------------------------------------------------------


@dataclass
class ClarificationContext:
    """Everything needed to resume a handler after the user's clarifying reply."""

    intent: str
    slots: dict                    # slots gathered so far
    missing_slot: str              # the slot we are still waiting for
    question: str                  # what was asked (for logging / re-ask)
    turn: int = 0                  # how many clarifying questions have been asked
    created_at: float = field(default_factory=time.monotonic)

    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > CLARIFICATION_TIMEOUT_SECONDS

    def turns_exhausted(self) -> bool:
        return self.turn >= MAX_CLARIFICATION_TURNS


# ---------------------------------------------------------------------------
# Per-satellite state store
# ---------------------------------------------------------------------------

# satellite IP → active clarification context (at most one per satellite)
_pending: dict[str, ClarificationContext] = {}


def set(satellite_ip: str, ctx: ClarificationContext) -> None:
    """Store a clarification context for a satellite, replacing any previous one."""
    _pending[satellite_ip] = ctx
    log.debug(
        "Clarification context set for %s: intent=%r missing_slot=%r turn=%d",
        satellite_ip, ctx.intent, ctx.missing_slot, ctx.turn,
    )


def get(satellite_ip: str) -> ClarificationContext | None:
    """
    Return the active clarification context for a satellite, or None.

    Returns None (and clears the entry) if the context has expired or
    exhausted its turn limit — the caller should then treat the incoming
    transcript as a fresh command.
    """
    ctx = _pending.get(satellite_ip)
    if ctx is None:
        return None

    if ctx.is_expired():
        log.info(
            "Clarification context for %s expired after %.1fs — clearing",
            satellite_ip, CLARIFICATION_TIMEOUT_SECONDS,
        )
        clear(satellite_ip)
        return None

    if ctx.turns_exhausted():
        log.info(
            "Clarification context for %s exhausted max turns (%d) — clearing",
            satellite_ip, MAX_CLARIFICATION_TURNS,
        )
        clear(satellite_ip)
        return None

    return ctx


def clear(satellite_ip: str) -> None:
    """Remove any pending clarification context for a satellite."""
    if _pending.pop(satellite_ip, None) is not None:
        log.debug("Clarification context cleared for %s", satellite_ip)


def increment_turn(satellite_ip: str) -> None:
    """Increment the turn counter for a satellite's active context."""
    ctx = _pending.get(satellite_ip)
    if ctx is not None:
        ctx.turn += 1
        log.debug(
            "Clarification turn %d/%d for %s",
            ctx.turn, MAX_CLARIFICATION_TURNS, satellite_ip,
        )

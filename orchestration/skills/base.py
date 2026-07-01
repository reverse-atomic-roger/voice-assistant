"""
skills/base.py

Shared types and reusable slot-parsing helpers for writing a skill.

A skill is a single file exposing one or more `Skill` objects — see
skills/README.md for the full author-facing guide. This module is normally
the only orchestrator-side thing a new skill needs to import:

    from skills.base import Skill, SlotSpec, ClarificationNeeded
    from skills.base import parse_value_string, parse_value_string_list

Skills also commonly import `database` directly for persistence, and only
rarely need `audio_io.send_canned` — only if a skill wants to suppress the
normal TTS response and play a pre-synthesised sound instead (see
skills/unknown.py for the one built-in example, and the audio_io module
docstring for why that import has to go the other way).
"""

from dataclasses import dataclass, field
from typing import Awaitable, Callable

# Re-exported purely for skill-author convenience — see ClarificationNeeded
# in conversation_state.py for the full field contract (intent, slots,
# missing_slot, question).
from conversation_state import ClarificationNeeded  # noqa: F401


@dataclass(frozen=True)
class SlotSpec:
    """
    Describes how to fill one slot during a clarification turn.

    description: plain-English instruction shown to the LLM in the
                  slot-fill prompt — describe what to extract and what JSON
                  shape to return it in. See the built-in skills for the
                  expected style (a short sentence plus a worked example).
    parse:        takes the raw fill_result dict the LLM returned and
                  produces the final value to store in `slots`. Must raise
                  ValueError on unusable input — the orchestrator treats
                  that as "re-ask the question" (up to conversation_state's
                  configured turn limit).
    """
    description: str
    parse: Callable[[dict], object]


@dataclass(frozen=True)
class Skill:
    """
    Everything the orchestrator needs to know about one intent.

    intent:       the intent name the LLM will return, e.g. "play_music".
                  Must be unique across all registered skills — checked at
                  startup by skills/registry.py.
    prompt_block: the chunk of the intent-extraction system prompt that
                  documents this intent and its slots to the LLM. Written
                  in the same indented style as the built-in skills (two
                  spaces before the intent name, four before each slot) —
                  see skills/timer.py for a worked example, and fold any
                  intent-specific formatting rules (e.g. "always return an
                  array") into this block rather than relying on a shared
                  rule elsewhere, since this block is the only thing that
                  travels with the skill.
    handler:      async def (slots: dict, satellite_ip: str) -> str | None.
                  Return a string to be spoken via TTS, or None if the
                  skill already handled all the audio output itself.
                  Raise ClarificationNeeded for a missing required slot —
                  the orchestrator asks the question and routes the reply
                  back through slot_specs below.
    slot_specs:   slot name -> SlotSpec, for any slot this skill might ask
                  the user to clarify. Keyed by plain slot name here — the
                  registry namespaces it to (intent, slot) automatically,
                  so slot names only need to be unique within this skill,
                  not across every skill ever written.
    """
    intent: str
    prompt_block: str
    handler: Callable[[dict, str], Awaitable[str | None]]
    slot_specs: dict[str, SlotSpec] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reusable slot-fill parsers
# ---------------------------------------------------------------------------
# Most slots a skill will ever need are either "a single string" or "a list
# of strings" — these two cover that, so a new skill rarely has to write its
# own parser. Only write a dedicated one when a slot needs real
# decomposition or validation, the way parse_duration does for durations.

def parse_value_string(fill_result: dict) -> str:
    """
    Generic parser for a single string slot. Expects {"value": "<string>"}.
    Raises ValueError if blank or missing.
    """
    value = str(fill_result.get("value", "")).strip()
    if not value:
        raise ValueError(f"Slot fill returned empty value: {fill_result!r}")
    return value


def parse_value_string_list(fill_result: dict) -> list[str]:
    """
    Generic parser for an array-of-strings slot. Expects
    {"value": [<string>, ...]}. A bare string is tolerated and treated as a
    single-item list rather than discarding an otherwise-usable reply.
    Raises ValueError if no usable items remain.
    """
    raw_items = fill_result.get("value", [])
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    elif not isinstance(raw_items, list):
        raw_items = []
    items = [str(item).strip() for item in raw_items if str(item).strip()]
    if not items:
        raise ValueError(f"Slot fill returned no usable items: {fill_result!r}")
    return items


def parse_duration(fill_result: dict) -> int:
    """
    Parser for a spoken duration. Per Deterministic-by-Default, the LLM is
    only ever asked to report the raw hours/minutes/seconds units the user
    said — never to multiply or add them. That arithmetic happens here, in
    plain Python, so a model arithmetic slip can never produce a wrong
    duration.
    Raises ValueError if the total is zero or negative.
    """
    def _as_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    hours = _as_int(fill_result.get("hours", 0))
    minutes = _as_int(fill_result.get("minutes", 0))
    seconds = _as_int(fill_result.get("seconds", 0))
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0:
        raise ValueError(
            f"Duration slot fill returned zero or negative total: {fill_result!r}"
        )
    return total

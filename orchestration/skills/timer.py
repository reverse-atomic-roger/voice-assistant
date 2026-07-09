"""
skills/timer.py

Built-in "timer" skill: set a countdown timer, optionally labelled.

Owns everything timer-shaped end to end, including what happens when the
countdown finishes: it schedules a generic core trigger via
`database.add_trigger(skill="timer", ...)` and registers `TRIGGER` below so
the orchestrator's trigger poller knows to call back into this module when
that row comes due. The core (orchestration.py, database.py) has no idea
what a "timer" is — see skills/base.py's TriggerHandler docstring for the
mechanism this relies on.
"""

import logging
from datetime import datetime, timedelta, timezone

import database
from skills.base import (
    ClarificationNeeded,
    Skill,
    SlotSpec,
    TriggerHandler,
    parse_duration,
    parse_value_string,
)

log = logging.getLogger(__name__)

PROMPT_BLOCK = """\
  timer
    hours    (integer, optional) hours component of the duration, if stated
    minutes  (integer, optional) minutes component of the duration, if stated
    seconds  (integer, optional) seconds component of the duration, if stated
    label    (string, optional)  short description of what the timer is for,
                                  only if the user actually gave one

    Extract each time unit exactly as the user stated it. Do NOT perform any
    arithmetic or unit conversion yourself — never convert "2 minutes" into
    120 seconds, never add units together. Just report the raw number for
    each unit mentioned; omit a unit (or use 0) if it was not mentioned.
    If the user did not give the timer a name, omit the label field entirely
    — do not invent one.
"""


def _normalize_duration(slots: dict) -> dict:
    """
    Deterministically convert raw hours/minutes/seconds slots (as supplied
    by the LLM) into a single duration_seconds total.

    If none of hours/minutes/seconds are present in `slots` — e.g. these
    slots already carry a duration_seconds total computed earlier via a
    completed clarification turn — the slots are returned unchanged.
    """
    if not any(key in slots for key in ("hours", "minutes", "seconds")):
        return slots

    slots = dict(slots)  # don't mutate the caller's dict

    def _as_int(value) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    hours = _as_int(slots.pop("hours", 0))
    minutes = _as_int(slots.pop("minutes", 0))
    seconds = _as_int(slots.pop("seconds", 0))

    slots["duration_seconds"] = hours * 3600 + minutes * 60 + seconds
    return slots


def _format_duration(duration_seconds: int) -> str:
    """
    Render a duration in seconds as a spoken phrase, e.g.
    "3 minutes", "1 hour, 30 minutes", "45 seconds".
    """
    minutes, seconds = divmod(duration_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    parts: list[str] = []
    if hours:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if seconds:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    return ", ".join(parts) if parts else "0 seconds"


async def handle(slots: dict, satellite_ip: str, target_satellites: list[str]) -> str | None:
    slots = _normalize_duration(slots)
    duration = int(slots.get("duration_seconds", 0) or 0)

    if duration <= 0:
        raise ClarificationNeeded(
            intent="timer",
            slots=slots,
            missing_slot="duration_seconds",
            question="Please specify duration.",
        )

    duration_str = _format_duration(duration)

    # No label given — no need to clarify. The duration itself doubles as
    # the label, e.g. an unlabelled ten-minute timer is just "10 minutes".
    label = slots.get("label", "")
    label = label.strip() if isinstance(label, str) else ""
    if not label:
        label = duration_str

    fires_at = datetime.now(timezone.utc) + timedelta(seconds=duration)

    # trigger_key doesn't need to be globally unique — it's only ever looked
    # up by this skill, scoped to skill="timer" — but the label makes for a
    # readable one and gives a future "cancel timer" handler something to
    # search on.
    database.add_trigger(
        skill="timer",
        trigger_key=label,
        fires_at=fires_at,
        origin_satellite_ip=satellite_ip,
        target_satellites=target_satellites,
        payload={"label": label},
    )

    log.info(
        "Timer set: label=%r duration=%ds fires_at=%s origin=%s targets=%s",
        label, duration, fires_at.isoformat(), satellite_ip, target_satellites,
    )

    return f"Timer set. {duration_str} remaining."


async def _on_timer_trigger(payload: dict) -> str:
    """
    Called by the orchestrator's trigger poller when a timer's trigger row
    comes due. `payload` is exactly what handle() stored above — the core
    never looks inside it, so this is the only place that needs to know a
    timer's payload shape is `{"label": str}`.
    """
    label = payload.get("label") or "Timer"
    return f"{label.capitalize()} timer complete."


TRIGGER = TriggerHandler(skill_name="timer", on_trigger=_on_timer_trigger)


PROMPT_BLOCK_CHECK_TIMER = """\
  check_timer
    label  (string, optional) which timer to check, only if the user named one

    Use this when the user asks how much time is left, whether a timer is
    still running, or to check on a timer — not to set a new one.
"""


async def handle_check_timer(slots: dict, satellite_ip: str, target_satellites: list[str]) -> str | None:
    label = slots.get("label", "")
    label = label.strip() if isinstance(label, str) else ""

    pending = database.get_pending_triggers(skill="timer")
    if not pending:
        return "You don't have any timers running."

    now = datetime.now(timezone.utc)

    if label:
        matches = [t for t in pending if label.lower() in t.payload.get("label", "").lower()]
        if not matches:
            others = ", ".join(t.payload.get("label", "a timer") for t in pending)
            return f"I couldn't find a timer called {label}. You have: {others}."
        pending = matches

    if len(pending) == 1:
        remaining = max(0, int((pending[0].fires_at - now).total_seconds()))
        timer_label = pending[0].payload.get("label", "Timer")
        return f"{timer_label.capitalize()} has {_format_duration(remaining)} left."

    parts = [
        f"{t.payload.get('label', 'a timer')}: {_format_duration(max(0, int((t.fires_at - now).total_seconds())))}"
        for t in pending
    ]
    return "You have multiple timers running. " + "; ".join(parts) + "."


SKILL_CHECK_TIMER = Skill(
    intent="check_timer",
    prompt_block=PROMPT_BLOCK_CHECK_TIMER,
    handler=handle_check_timer,
)


SKILL = Skill(
    intent="timer",
    prompt_block=PROMPT_BLOCK,
    handler=handle,
    slot_specs={
        "duration_seconds": SlotSpec(
            description=(
                "The duration of the timer as stated by the user. "
                "Extract hours, minutes, and seconds as separate integer fields. "
                "Return a JSON object with keys: hours (int), minutes (int), seconds (int). "
                "Use 0 for any unit not mentioned. Example: '3 minutes' → "
                '{"hours": 0, "minutes": 3, "seconds": 0}'
            ),
            parse=parse_duration,
        ),
        "label": SlotSpec(
            description=(
                "A short description of what the timer is for. "
                "Return a JSON object with key: value (string). "
                "Example: 'the pasta' → {\"value\": \"pasta\"}"
            ),
            parse=parse_value_string,
        ),
    },
)

# The registry (skills/registry.py) reads SKILLS, not SKILL, when it builds
# the flat REGISTERED_SKILLS list — a one-intent module like this still
# needs the list, even though it only has one element, so that adding a
# multi-intent skill elsewhere doesn't require a different convention.
# check_timer is optional-slot-only (no ClarificationNeeded path), so it
# needs no entry of its own in any slot_specs dict.
SKILLS = [SKILL, SKILL_CHECK_TIMER]

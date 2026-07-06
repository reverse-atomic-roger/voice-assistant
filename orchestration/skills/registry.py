"""
skills/registry.py

The single place to enable a skill. To add one:
    1. Write skills/your_skill.py, exposing a module-level Skill (or several).
    2. Import it below.
    3. Add it to REGISTERED_SKILLS.
    4. If it schedules future events, add it to SKILL_MODULES too (see
       "Trigger handlers" below) and expose a module-level TRIGGER.

That's the whole job — orchestration.py builds its handler table, its
intent-extraction prompt, its slot-fill registry, and its trigger-handler
table from this file at import time. Nothing else in the codebase needs to
change.

This is a deliberately explicit, hand-edited list rather than directory
auto-scanning. A skill is arbitrary Python with full access to the database
and the satellite network, so enabling one should be a conscious, auditable
action — read the file, then add the line — the same instinct as reading a
PKGBUILD before you build it, not auto-running whatever happens to land in
a folder.

Trigger handlers
-----------------
Any registered skill module can additionally expose a module-level
`TRIGGER = TriggerHandler(...)` (see skills/base.py) if it needs to do
something at a future time — a countdown, a scheduled announcement, etc.
skills/timer.py is the one built-in example. This registry scans
SKILL_MODULES for that attribute and builds TRIGGER_HANDLERS from whatever
it finds; a skill with no TRIGGER attribute is simply skipped, so most
skills never need to think about this at all.
"""

import logging

from skills import lists, timer, unknown
from skills.base import Skill, TriggerHandler

# To enable a shared/community skill:
#   from skills import play_music
# ...then add play_music.SKILL to REGISTERED_SKILLS below, and play_music
# to SKILL_MODULES too if it exposes a TRIGGER.

log = logging.getLogger(__name__)

REGISTERED_SKILLS: list[Skill] = [
    timer.SKILL,
    lists.SKILL_LIST_ADD,
    lists.SKILL_LIST_READ,
    lists.SKILL_LIST_CLEAR,
    lists.SKILL_LIST_MERGE,
    unknown.SKILL,
]

# Every module that backs a registered skill, scanned below for an optional
# module-level TRIGGER. Listed separately from REGISTERED_SKILLS because a
# module can expose several Skill objects (see lists.py) but at most one
# TRIGGER — modules, not intents, are what own a trigger handler.
SKILL_MODULES = [timer, lists, unknown]


def _validate(skills: list[Skill]) -> None:
    """
    Sanity-check the registry at import time. Fails loudly and immediately
    rather than letting a broken registration surface as a confusing
    runtime error mid-conversation.
    """
    seen_intents: set[str] = set()
    for skill in skills:
        if not skill.intent:
            raise RuntimeError(f"Skill has an empty intent name: {skill!r}")
        if skill.intent in seen_intents:
            raise RuntimeError(
                f"Duplicate intent {skill.intent!r} registered more than once "
                f"in REGISTERED_SKILLS — intent names must be unique."
            )
        seen_intents.add(skill.intent)

        if not callable(skill.handler):
            raise RuntimeError(f"Skill {skill.intent!r} has a non-callable handler")

        if not skill.prompt_block.strip():
            raise RuntimeError(f"Skill {skill.intent!r} has an empty prompt_block")

    # orchestration.dispatch() falls back to the "unknown" handler for any
    # intent name it doesn't recognise — that fallback only works if
    # "unknown" itself is actually registered.
    if "unknown" not in seen_intents:
        raise RuntimeError(
            "No skill registered for intent 'unknown' — dispatch's fallback "
            "handler depends on it being present. Don't remove skills/unknown.py "
            "from REGISTERED_SKILLS."
        )

    log.info("Loaded %d skill(s): %s", len(skills), sorted(seen_intents))


def _collect_trigger_handlers(modules: list) -> dict[str, TriggerHandler]:
    """
    Scan `modules` for an optional module-level TRIGGER and build the
    skill-name -> TriggerHandler table the orchestrator's trigger poller
    uses to route a fired trigger back to the skill that scheduled it.

    Fails loudly (same philosophy as _validate above) on a malformed
    TRIGGER attribute or a duplicate skill_name, rather than letting a
    typo silently mean "this skill's timers never fire."
    """
    handlers: dict[str, TriggerHandler] = {}
    for module in modules:
        trigger = getattr(module, "TRIGGER", None)
        if trigger is None:
            continue

        if not isinstance(trigger, TriggerHandler):
            raise RuntimeError(
                f"{module.__name__}.TRIGGER must be a TriggerHandler, got {trigger!r}"
            )
        if not trigger.skill_name:
            raise RuntimeError(f"{module.__name__}.TRIGGER has an empty skill_name")
        if trigger.skill_name in handlers:
            raise RuntimeError(
                f"Duplicate TriggerHandler skill_name {trigger.skill_name!r} — "
                f"registered by both {module.__name__!r} and another module. "
                f"skill_name must be unique."
            )
        if not callable(trigger.on_trigger):
            raise RuntimeError(
                f"{module.__name__}.TRIGGER has a non-callable on_trigger"
            )

        handlers[trigger.skill_name] = trigger

    log.info("Loaded %d trigger handler(s): %s", len(handlers), sorted(handlers))
    return handlers


_validate(REGISTERED_SKILLS)

TRIGGER_HANDLERS: dict[str, TriggerHandler] = _collect_trigger_handlers(SKILL_MODULES)

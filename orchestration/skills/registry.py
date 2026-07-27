"""
skills/registry.py

The single place to enable a skill. To add one:
    1. Write skills/your_skill.py, exposing a module-level SKILLS list —
       every intent your skill handles, even if there's only one (see
       skills/README.md for the per-module contract).
    2. Add the module to SKILL_MODULES below. That's it — one line, one
       import, regardless of whether your skill has one intent or fifteen.

SKILL_MODULES is the only hand-maintained list in this file. Everything
else — the flat REGISTERED_SKILLS list orchestration.py dispatches on, the
intent-extraction prompt, the slot-fill registry, and the trigger-handler
table — is derived from it at import time. A five-intent skill (play,
pause, skip, queue, playlist, say) is still exactly one entry here, the
same as a one-intent skill like timer.

This is a deliberately explicit, hand-edited list rather than directory
auto-scanning. A skill is arbitrary Python with full access to the database
and the satellite network, so enabling one should be a conscious, auditable
action — read the file, then add the line — the same instinct as reading a
PKGBUILD before you build it, not auto-running whatever happens to land in
a folder.

Trigger handlers
-----------------
Any module in SKILL_MODULES can additionally expose a module-level
`TRIGGER = TriggerHandler(...)` (see skills/base.py) if it needs to do
something at a future time — a countdown, a scheduled announcement, etc.
skills/timer.py is the one built-in example. This registry scans
SKILL_MODULES for that attribute and builds TRIGGER_HANDLERS from whatever
it finds; a skill with no TRIGGER attribute is simply skipped, so most
skills never need to think about this at all.
"""

import inspect
import logging

from skills import lists, timer, music, unknown
from skills.base import Skill, TriggerHandler

# To enable a shared/community skill:
#   from skills import play_music
# ...then add play_music to the list below. Its whole SKILLS list (however
# many intents it defines) and its TRIGGER (if any) come along for free.
SKILL_MODULES = [timer, lists, music, unknown]

log = logging.getLogger(__name__)


def _collect_registered_skills(modules: list) -> list[Skill]:
    """
    Flatten every module's SKILLS list into the one master list
    orchestration.py dispatches on. This is the only place per-intent Skill
    objects get assembled — a module contributes as many intents as it
    likes by putting them all in one SKILLS list, so adding a five-intent
    skill to the assistant is one line in SKILL_MODULES, not five lines
    here.
    """
    collected: list[Skill] = []
    for module in modules:
        module_skills = getattr(module, "SKILLS", None)
        if module_skills is None:
            raise RuntimeError(
                f"{module.__name__} is listed in SKILL_MODULES but has no "
                f"module-level SKILLS list. Every skill module must expose "
                f"SKILLS: list[Skill], even if it only defines one intent — "
                f"see skills/README.md."
            )
        if not isinstance(module_skills, list) or not module_skills:
            raise RuntimeError(
                f"{module.__name__}.SKILLS must be a non-empty list of Skill objects, "
                f"got {module_skills!r}"
            )
        collected.extend(module_skills)
    return collected


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
                f"Duplicate intent {skill.intent!r} registered more than once — "
                f"intent names must be unique across every module in SKILL_MODULES."
            )
        seen_intents.add(skill.intent)

        if not callable(skill.handler):
            raise RuntimeError(f"Skill {skill.intent!r} has a non-callable handler")

        # Every handler must accept exactly (slots, satellite_ip,
        # target_satellites, user_id) — checked once here, at import time,
        # rather than on every dispatch() call, so a skill author who
        # forgets to add the user_id parameter gets a loud failure at
        # startup instead of a silent TypeError mid-conversation.
        handler_params = inspect.signature(skill.handler).parameters
        if len(handler_params) != 4:
            raise RuntimeError(
                f"Skill {skill.intent!r} handler must accept exactly 4 parameters "
                f"(slots, satellite_ip, target_satellites, user_id), got "
                f"{len(handler_params)}: {list(handler_params)}"
            )

        if not skill.prompt_block.strip():
            raise RuntimeError(f"Skill {skill.intent!r} has an empty prompt_block")

    # orchestration.dispatch() falls back to the "unknown" handler for any
    # intent name it doesn't recognise — that fallback only works if
    # "unknown" itself is actually registered.
    if "unknown" not in seen_intents:
        raise RuntimeError(
            "No skill registered for intent 'unknown' — dispatch's fallback "
            "handler depends on it being present. Don't remove skills/unknown "
            "from SKILL_MODULES."
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

        # Same one-time arity check as handler above — (payload, user_id).
        trigger_params = inspect.signature(trigger.on_trigger).parameters
        if len(trigger_params) != 2:
            raise RuntimeError(
                f"{module.__name__}.TRIGGER.on_trigger must accept exactly 2 "
                f"parameters (payload, user_id), got {len(trigger_params)}: "
                f"{list(trigger_params)}"
            )

        handlers[trigger.skill_name] = trigger

    log.info("Loaded %d trigger handler(s): %s", len(handlers), sorted(handlers))
    return handlers


REGISTERED_SKILLS: list[Skill] = _collect_registered_skills(SKILL_MODULES)

_validate(REGISTERED_SKILLS)

TRIGGER_HANDLERS: dict[str, TriggerHandler] = _collect_trigger_handlers(SKILL_MODULES)

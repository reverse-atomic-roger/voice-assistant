"""
skills/registry.py

The single place to enable a skill. To add one:
    1. Write skills/your_skill.py, exposing a module-level Skill (or several).
    2. Import it below.
    3. Add it to REGISTERED_SKILLS.

That's the whole job — orchestration.py builds its handler table, its
intent-extraction prompt, and its slot-fill registry from this list at
import time. Nothing else in the codebase needs to change.

This is a deliberately explicit, hand-edited list rather than directory
auto-scanning. A skill is arbitrary Python with full access to the database
and the satellite network, so enabling one should be a conscious, auditable
action — read the file, then add the line — the same instinct as reading a
PKGBUILD before you build it, not auto-running whatever happens to land in
a folder.
"""

import logging

from skills import converse, lists, timer, unknown
from skills.base import Skill

# To enable a shared/community skill:
#   from skills import play_music
# ...then add play_music.SKILL to the list below.

log = logging.getLogger(__name__)

REGISTERED_SKILLS: list[Skill] = [
    timer.SKILL,
    lists.SKILL_LIST_ADD,
    lists.SKILL_LIST_READ,
    lists.SKILL_LIST_CLEAR,
    lists.SKILL_LIST_MERGE,
    converse.SKILL,
    unknown.SKILL,
]


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


_validate(REGISTERED_SKILLS)

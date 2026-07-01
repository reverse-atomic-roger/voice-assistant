"""
skills/converse.py

Stub for escalating conversational/ambiguous requests to a larger model.
Not yet implemented — placeholder so the intent has somewhere to land.
"""

import logging

from skills.base import Skill

log = logging.getLogger(__name__)

PROMPT_BLOCK = """\
  converse
    (no slots — use when the request is conversational, ambiguous, or
     requires reasoning rather than a simple action)
"""


async def handle(slots: dict, satellite_ip: str) -> str | None:
    log.info("STUB converse: escalation to large model not yet implemented")
    # TODO: call large Ollama model with full conversation context
    return "That function is not yet available."


SKILL = Skill(intent="converse", prompt_block=PROMPT_BLOCK, handler=handle)

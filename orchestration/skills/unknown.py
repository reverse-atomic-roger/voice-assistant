"""
skills/unknown.py

Fallback skill for when the intent model doesn't recognise the request.
Dispatch also falls back to this skill's handler directly if an intent name
comes back that isn't registered at all — see orchestration.dispatch().
"""

import logging

import audio_io
from skills.base import Skill

log = logging.getLogger(__name__)

PROMPT_BLOCK = """\
  unknown
    (no slots — use when the request does not match any supported intent)
"""


async def handle(slots: dict, satellite_ip: str) -> str | None:
    log.info("Unknown intent — sending canned response")
    await audio_io.send_canned("unknown", satellite_ip)
    return None  # canned audio already sent, no TTS needed


SKILL = Skill(intent="unknown", prompt_block=PROMPT_BLOCK, handler=handle)

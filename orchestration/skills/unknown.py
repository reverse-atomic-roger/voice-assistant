"""
skills/unknown.py

Fallback skill for when the intent model doesn't recognise the request.
Dispatch also falls back to this skill's handler directly if an intent name
comes back that isn't registered at all — see orchestration.dispatch().

Deliberately ignores target_satellites: an "I didn't understand that"
response should always stay with whoever spoke, never get broadcast to a
room they may have named for a *different* part of the (misunderstood)
request. See voice-assistant-refactor-2026-07-02.md, Part 2, "Why this
design, and not something else" for the reasoning.
"""

import logging

import audio_io
from skills.base import Skill

log = logging.getLogger(__name__)

PROMPT_BLOCK = """\
  unknown
    (no slots — use when the request does not match any supported intent)
"""


async def handle(slots: dict, satellite_ip: str, target_satellites: list[str]) -> str | None:
    log.info("Unknown intent — sending canned response")
    await audio_io.send_canned("unknown", satellite_ip)
    return None  # canned audio already sent, no TTS needed


SKILL = Skill(intent="unknown", prompt_block=PROMPT_BLOCK, handler=handle)

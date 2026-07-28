"""
skills/locate.py

Built-in "locate" skill: locate_user ("Computer, locate Alice.", "Where am
I?", "Where's Alice?").

This is a thin read-only front end over database.py's presence tables —
all the actual work (BLE sighting ingestion, hysteresis, staleness) is core
infrastructure this module has no opinion about. It exists because
"report where someone is" and "silently follow someone's music between
rooms" (see skills/music.py) are different-enough concerns that they don't
belong in the same handler, even though both read the exact same
get_user_location() this module calls.

No mismatch guard here (contrast with skills/music.py's
_maybe_start_follow_session, which calls database.check_user_at_satellite
before trusting user_id enough to bind a session to it) — locate_user
doesn't bind or persist anything, it just reads and speaks back whatever
presence currently resolves to, so there's nothing for a mismatch to
corrupt.

Name matching against presence data is deliberately case-insensitive
(user_id as produced by the STT server's speaker-ID step is assumed to be
a lowercase-normalised name, e.g. "alice") since a spoken request may
capitalise it however the small intent model feels like. If your speaker
profiles use a different id scheme (not a plain first name), the
_display_name() capitalisation below will look wrong even though lookups
still work correctly — that's a cosmetic fix, not a functional one.
"""

import logging

from skills.base import ClarificationNeeded, Skill, SlotSpec, parse_value_string

import database

log = logging.getLogger(__name__)

# Words the small intent model might reasonably put in target_user when the
# person is asking about themselves rather than naming someone else.
_SELF_REFERENCES = {"me", "myself", "i", ""}

PROMPT_BLOCK = """\
  locate_user
    target_user  (string) who to locate. Put the name exactly as the user
                 said it (e.g. "Alice"). If the user is asking about
                 themselves rather than naming someone else — "where am
                 I", "locate me", "am I in the kitchen" — put "me" instead
                 of guessing a name.
"""


async def handle(
    slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str,
) -> str | None:
    target_user = str(slots.get("target_user", "") or "").strip()

    if target_user.lower() in _SELF_REFERENCES:
        if user_id == "unknown":
            return "I don't know who's asking, so I can't locate you."
        lookup_id = user_id
        display_name = "You"
        is_self = True
    else:
        lookup_id = target_user.lower()
        display_name = target_user
        is_self = False

    if not lookup_id:
        raise ClarificationNeeded(
            intent="locate_user",
            slots=slots,
            missing_slot="target_user",
            question="Who would you like me to locate?",
        )

    location_ip = database.get_user_location(lookup_id)
    if location_ip is None:
        log.info("locate_user: no presence data for %r", lookup_id)
        if is_self:
            return "I don't have a location for you."
        return f"I have no location for {display_name}."

    room = database.satellite_name(location_ip)
    log.info("locate_user: %r resolved to %s", lookup_id, room)

    if is_self:
        return f"You are in the {room}."
    return f"{display_name} is in the {room}."


SKILL_LOCATE_USER = Skill(
    intent="locate_user",
    prompt_block=PROMPT_BLOCK,
    handler=handle,
    router_hint=(
        "Find out what room a person is currently in, including asking "
        "where the speaker themself currently is."
    ),
    slot_specs={
        "target_user": SlotSpec(
            description=(
                'Who to locate. Use "me" if the user is asking about '
                "themselves rather than naming someone else. "
                "Return a JSON object with key: value (string)."
            ),
            parse=parse_value_string,
        ),
    },
)

SKILLS = [SKILL_LOCATE_USER]

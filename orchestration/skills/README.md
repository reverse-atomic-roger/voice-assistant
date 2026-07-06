# Writing a skill

A skill is one Python file that teaches the assistant a new intent —
"play a song", "search the logs", whatever you need. Once it's written,
enabling it is two lines in `skills/registry.py`. Nothing else in the
orchestrator needs to change.

## The contract

A skill is one or more module-level `Skill` objects:

```python
from skills.base import Skill, SlotSpec, ClarificationNeeded, parse_value_string

PROMPT_BLOCK = """\
  play_music
    song_title  (string) the song the user wants to play
"""

async def handle(slots: dict, satellite_ip: str, target_satellites: list[str]) -> str | None:
    song = slots.get("song_title", "").strip()
    if not song:
        raise ClarificationNeeded(
            intent="play_music",
            slots=slots,
            missing_slot="song_title",
            question="What song?",
        )
    # ... do the thing ...
    return f"Playing {song}."

SKILL = Skill(
    intent="play_music",
    prompt_block=PROMPT_BLOCK,
    handler=handle,
    slot_specs={
        "song_title": SlotSpec(
            description=(
                "The song the user wants to play. "
                "Return a JSON object with key: value (string)."
            ),
            parse=parse_value_string,
        ),
    },
)
```

That's the whole interface. Three pieces:

- **`prompt_block`** — tells the small intent-extraction LLM your intent
  exists and what slots it has. Two-space indent for the intent name,
  four-space for each slot, same style as the built-in skills
  (`skills/timer.py` is a good example with optional slots and an
  intent-specific instruction paragraph). Fold any formatting rules
  specific to your intent into this block — don't assume anything outside
  your skill's own file.
- **`handler`** — `async def (slots: dict, satellite_ip: str, target_satellites: list[str]) -> str | None`.
  Return a string to be spoken back to the user, or `None` if you already
  handled the audio yourself. Raise `ClarificationNeeded` for any required
  slot that's missing — the orchestrator will ask the question for you and
  bring the answer back through `slot_specs`.
  `satellite_ip` is the satellite that heard the command (the origin) —
  use it for anything that should always go back to the speaker regardless
  of routing. `target_satellites` is where the orchestrator will actually
  deliver your returned text, already resolved from whatever the user
  said (e.g. "in the kitchen"), defaulting to `[satellite_ip]` if they
  named nowhere. Most skills can just return text and ignore
  `target_satellites` entirely — the orchestrator handles delivery to
  every target for you. Only reach for it directly if your skill needs
  the destination for something else, like passing it through to
  `database.add_trigger(...)` the way `skills/timer.py` does.
- **`slot_specs`** — only needed for slots you might ask the user to
  clarify. For a plain string slot, `parse_value_string` covers it; for
  "one or more items", `parse_value_string_list` covers it. Write your own
  parser only if the slot needs real decomposition (see `parse_duration` in
  `skills/base.py` for why timer durations get one: the LLM is never asked
  to do arithmetic, only to report the raw units the user said).

## Registering it

In `skills/registry.py`:

```python
from skills import play_music   # 1. import your module

REGISTERED_SKILLS: list[Skill] = [
    ...,
    play_music.SKILL,            # 2. add it to the list
]
```

The registry validates itself at import time — a duplicate intent name, a
missing handler, or an empty prompt block will fail loudly at startup
rather than misbehaving mid-conversation.

## What you get for free

- `database` — for persistence, same as the built-in skills use.
- Clarification handling — your handler doesn't need to know or care
  whether a slot came from the first request or a follow-up reply, and a
  room named before a clarifying question ("set a timer in the kitchen for
  ... uh...") is preserved through the follow-up automatically.
- Multi-room routing — "in the kitchen", "in the bedroom and the kitchen",
  or nothing at all (defaults to wherever the command was heard) is parsed
  and resolved to satellite IPs before your handler ever runs. Just return
  text; the orchestrator delivers it everywhere it needs to go. See
  `target_satellites` in "The contract" above if your skill needs to know
  the destination itself.
- Slot names only need to be unique *within your skill* — the registry
  keys everything by `(intent, slot)`, so two skills can both have a
  `name` slot without colliding.

## Owning your own persistence

If your skill needs to remember something between requests, it gets its
own table — you don't add a function to `database.py` for it, and
`database.py` never needs to change. Two calls:

```python
import database

_SCHEMA = """
CREATE TABLE IF NOT EXISTS play_music_favorites (
    id      INTEGER PRIMARY KEY,
    song    TEXT NOT NULL
);
"""
database.register_schema(_SCHEMA)   # once, at module level

def _add_favorite(song: str) -> None:
    conn = database.get_connection()
    conn.execute("INSERT INTO play_music_favorites (song) VALUES (?)", (song,))
    conn.commit()
```

`register_schema()` queues your DDL to be applied alongside every other
skill's when the orchestrator calls `database.init()` at startup.
`get_connection()` hands back the same shared SQLite connection every
built-in skill already uses — write whatever queries your table needs,
same as `skills/lists.py` does for its `lists`/`list_items` tables.

A couple of things to keep in mind:

- Use `CREATE TABLE IF NOT EXISTS` / `CREATE INDEX IF NOT EXISTS` — your
  schema runs every time the orchestrator starts.
- `register_schema()` has to run at import time (module level, like the
  example above), not from inside a handler — it raises if called after
  `database.init()` has already run.
- This is for storage nothing else touches. If two skills need to share
  state, or the orchestrator itself needs to query it directly outside of
  any handler (timers are the one built-in example — the timer poller in
  `orchestration.py` reads them directly), that's a real function in
  `database.py`, not a skill-owned table.

## Scheduling future events (triggers)

If your skill needs to do something later — a countdown, a scheduled
announcement, anything with a "fires at time T" shape — you don't add
polling logic to orchestration.py. There's already one generic poller
shared by every skill; you just hook into it.

Two calls:

```python
import database
from skills.base import Skill, TriggerHandler

async def handle(slots: dict, satellite_ip: str, target_satellites: list[str]) -> str | None:
    # ... figure out when this should fire and what to remember ...
    database.add_trigger(
        skill="play_music",              # matches TRIGGER.skill_name below
        trigger_key=song,                # skill-local, not interpreted by the core
        fires_at=fires_at,               # UTC-aware datetime
        origin_satellite_ip=satellite_ip,      # who asked, for error reporting
        target_satellites=target_satellites,   # where the announcement plays
        payload={"song": song},          # whatever your on_trigger callback needs
    )
    return "Got it, I'll let you know."

async def _on_fire(payload: dict) -> str | None:
    return f"{payload['song']} finished."

TRIGGER = TriggerHandler(skill_name="play_music", on_trigger=_on_fire)
```

Then add your module to `SKILL_MODULES` in `skills/registry.py` (alongside
adding your `Skill` objects to `REGISTERED_SKILLS` as usual). That's the
whole job — the registry picks up your `TRIGGER` automatically from there.

A few things worth knowing about how this works:

- **`payload` is entirely yours.** The core stores and returns it as opaque
  JSON — it never reads or validates its contents. Put whatever your
  `on_trigger` callback needs to compose its announcement.
- **Routing is decided once, at scheduling time, not part of payload.**
  `origin_satellite_ip` and `target_satellites` are real arguments to
  `add_trigger`, not something you smuggle through your own payload —
  routing audio is core delivery infrastructure. Just pass through the
  `satellite_ip` and `target_satellites` your `handle()` was called with;
  by the time the trigger fires, the core already knows who to tell if
  delivery fails (`origin_satellite_ip`) and where to actually play the
  announcement (`target_satellites`), without `on_trigger` needing to
  decide either one again.
- **`on_trigger` returns text or `None`.** Return the string to speak, or
  `None` if your skill already handled its own audio output and there's
  nothing more to say.
- **One `TRIGGER` per module.** If a module exposes several `Skill` objects
  (like `lists.py` does) but only some of them schedule future events, it
  still only needs one `TRIGGER` — `skill=` at scheduling time is what ties
  a specific trigger row back to it, not the intent that created it.
- **`skill_name` must be unique**, checked the same way `intent` is —
  `skills/registry.py` fails loudly at import time on a collision.

## One thing to know before sharing a skill

A skill runs with the same access as the rest of the orchestrator — your
database, your satellites. If you're installing someone else's skill file,
read it first, the same way you'd read a shell script or a PKGBUILD before
running it. Nothing in this codebase will silently auto-load a skill on
your behalf, by design.

## A note on scale

Every registered skill's `prompt_block` gets concatenated into the *same*
system prompt sent to the small intent model on every utterance.
This works comfortably for a handful of skills. If you end up with a large
number installed at once, a small local model's intent accuracy may start
to degrade simply from having more options to choose between. Consider a 
larger or more focused model for intent extraction if accuracy drops below
acceptable levels
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

async def handle(slots: dict, satellite_ip: str) -> str | None:
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
- **`handler`** — `async def (slots: dict, satellite_ip: str) -> str | None`.
  Return a string to be spoken back to the user, or `None` if you already
  handled the audio yourself. Raise `ClarificationNeeded` for any required
  slot that's missing — the orchestrator will ask the question for you and
  bring the answer back through `slot_specs`.
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
  whether a slot came from the first request or a follow-up reply.
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
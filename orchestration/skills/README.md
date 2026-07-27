# Writing a skill

A skill is one Python file that teaches the assistant one or more new
intents — "play a song", "search the logs", a whole family like "play",
"pause", "skip", "queue" for a music skill. Once it's written, enabling the
entire module — however many intents it defines — is one line in
`skills/registry.py`. Nothing else in the orchestrator needs to change.

## The contract

A skill is one or more module-level `Skill` objects, collected into one
module-level `SKILLS` list — that list, not the individual `Skill` objects,
is what `skills/registry.py` reads:

```python
from skills.base import Skill, SlotSpec, ClarificationNeeded, parse_value_string

PROMPT_BLOCK = """\
  play_music
    song_title  (string) the song the user wants to play
"""

async def handle(slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None:
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

SKILL_PLAY = Skill(
    intent="play_music",
    prompt_block=PROMPT_BLOCK,
    handler=handle,
    router_hint="Play a song by name.",
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

# Every intent this module handles, gathered in one place. A one-intent
# skill still needs this — it's just a one-element list — so the
# convention stays the same whether your module has one intent or ten.
# Add "pause_music", "skip_music", etc. later and this list is the only
# thing in this file that grows; skills/registry.py doesn't change at all.
SKILLS = [SKILL_PLAY]
```

That's the whole interface. Five pieces:

- **`prompt_block`** — tells the small intent-extraction LLM your intent
  exists and what slots it has. Two-space indent for the intent name,
  four-space for each slot, same style as the built-in skills
  (`skills/timer.py` is a good example with optional slots and an
  intent-specific instruction paragraph). Fold any formatting rules
  specific to your intent into this block — don't assume anything outside
  your skill's own file.
- **`handler`** — `async def (slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None`.
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
  `user_id` is the speaker identified by the STT server's speaker-ID step
  (`"unknown"` if unidentified, or below its confidence threshold). Every
  handler must accept it, even if it never looks at the value — the
  registry checks your handler's parameter count at import time and fails
  loudly if it doesn't match, the same way it fails on a duplicate intent
  name. Most skills can ignore it entirely, same as `target_satellites`;
  only use it if behaviour or persisted data should genuinely vary by who's
  asking (`skills/timer.py` does this — a timer remembers who set it, and
  its completion announcement uses that name when known).
- **`router_hint`** — one plain-English sentence describing what the
  intent is for, e.g. `"Play a song, artist, album, or mood/vibe
  description."` Used by `intent_router.py` to decide which skills'
  `prompt_block`s are worth sending to the LLM on a given request as the
  number of registered skills grows — see "A note on scale" below. Not
  shown to the extraction LLM and has no formatting rules; write it the
  way you'd describe the intent to a person. Optional, but skipping it
  means weaker routing for your skill (a startup warning will say so).
- **`slot_specs`** — only needed for slots you might ask the user to
  clarify. For a plain string slot, `parse_value_string` covers it; for
  "one or more items", `parse_value_string_list` covers it. Write your own
  parser only if the slot needs real decomposition (see `parse_duration` in
  `skills/base.py` for why timer durations get one: the LLM is never asked
  to do arithmetic, only to report the raw units the user said).
- **`SKILLS`** — the module-level list of every `Skill` this file defines.
  Required even for a single-intent module (`SKILLS = [SKILL]`) — this is
  what `skills/registry.py` actually reads, so the registration story is
  identical whether your module has one intent or ten.

## Registering it

In `skills/registry.py`, add your module to `SKILL_MODULES`:

```python
from skills import lists, timer, unknown, play_music   # 1. import your module

SKILL_MODULES = [timer, lists, unknown, play_music]     # 2. add it here
```

That's it — one line, regardless of how many intents `play_music.SKILLS`
contains. `skills/registry.py` flattens every module's `SKILLS` list into
the master list orchestration.py dispatches on, so a five-intent music
skill (play, pause, skip, queue, playlist) is exactly as much work to
register as a one-intent skill like timer.

The registry validates itself at import time — a duplicate intent name, a
missing handler, a handler (or `TRIGGER.on_trigger`) with the wrong number
of parameters, an empty prompt block, or a module in `SKILL_MODULES` that
forgot to define `SKILLS` will all fail loudly at startup rather than
misbehaving mid-conversation.

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
- Speaker identification — the STT server resolves the speaker against
  enrolled voice profiles before your handler ever runs, and the result
  arrives as `user_id` (`"unknown"` if unidentified or below its confidence
  threshold). No skill needs to do its own speaker matching; just read
  `user_id` if your skill's behaviour should vary by who's asking.
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

async def handle(slots: dict, satellite_ip: str, target_satellites: list[str], user_id: str) -> str | None:
    # ... figure out when this should fire and what to remember ...
    database.add_trigger(
        skill="play_music",              # matches TRIGGER.skill_name below
        trigger_key=song,                # skill-local, not interpreted by the core
        fires_at=fires_at,               # UTC-aware datetime
        origin_satellite_ip=satellite_ip,      # who asked, for error reporting
        target_satellites=target_satellites,   # where the announcement plays
        payload={"song": song},          # whatever your on_trigger callback needs
        user_id=user_id,                 # who asked, for ownership/personalisation
    )
    return "Got it, I'll let you know."

async def _on_fire(payload: dict, user_id: str) -> str | None:
    return f"{payload['song']} finished."

TRIGGER = TriggerHandler(skill_name="play_music", on_trigger=_on_fire)
```

Then add your module to `SKILL_MODULES` in `skills/registry.py`, same as
any other skill (see "Registering it" above) — there's no separate step
for the trigger side of things. That's the whole job — the registry picks
up your `TRIGGER` automatically from there.

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
- **Ownership is stored the same way.** `add_trigger`'s `user_id` argument
  works exactly like `origin_satellite_ip` — pass through the `user_id`
  your `handle()` was called with, and it comes back to `on_trigger` when
  the trigger fires (`"unknown"` if the speaker wasn't identified). Use it
  if your announcement should address a specific person by name; ignore it
  if your skill has no concept of ownership.
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

Every registered skill's `prompt_block` is a *candidate* for the system
prompt sent to the small intent model — but not every skill's block is
necessarily included on every request. `skills/registry.py` still builds
the full list `orchestration.py` could dispatch on, but before a
transcript reaches the LLM, `intent_router.py` pre-filters that list down
to the handful of skills whose purpose is closest in meaning to what was
just said, and only sends *their* `prompt_block`s. This is what keeps a
growing skill collection from degrading a small local model's intent
accuracy simply from having more options to choose between on every
single request.

**Set `router_hint` on every `Skill` you write** (see "The contract"
above) — the router embeds it once at startup and compares it against the
embedded transcript at request time, using the same local embedding model
`skills/music.py` already uses for its mood search
(`nomic-embed-text`, via Ollama). Skip it and your skill still works, just
with a weaker fallback signal (its own intent name) — the router logs a
warning at startup so the gap doesn't go unnoticed.

If your module defines several closely-related intents — `play_music`,
`pause_music`, `set_volume` all belonging to one music skill, say — write
`router_hint`s that read as related to each other (all mentioning "music
playback", for instance). The router's job is finding "requests that are
probably about the same kind of thing"; hints that don't share vocabulary
with their own siblings can get shortlisted apart from each other even
though a user is likely to move between them in the same conversation
("play some jazz" → "turn it up" → "pause it").

You don't need to worry about the router excluding your skill by mistake
and breaking it outright: it's deliberately biased toward over-including
rather than under-including a candidate. It returns a generous number of
matches, always includes `unknown` regardless of score, and falls back to
sending *every* registered skill's `prompt_block` — today's un-filtered
behaviour — whenever it isn't confident about a given transcript (a low
similarity score, an Ollama hiccup, or the router never having built
successfully at startup). A missing or vague `router_hint` degrades
routing quality; it can't take your skill down. See `intent_router.py`'s
module docstring for the full mechanics, and its `MIN_CONFIDENT_SIMILARITY`
and `SHORTLIST_SIZE` constants if you're tuning it against real traffic.
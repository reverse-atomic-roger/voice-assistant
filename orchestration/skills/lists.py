"""
skills/lists.py

Built-in list skills: "list_add", "list_read", "list_clear", and
"list_merge". They share storage, a "list name" concept, and a join-items
helper, so they live together in one file with several Skill objects
rather than being artificially split into one file per intent — a skill
module is the unit of *sharing*, not a hard one-file-per-intent rule.

This module owns its own persistence rather than calling into database.py:
the `lists` / `list_items` tables and every query against them live right
here, registered via `database.register_schema()` and run through
`database.get_connection()`. Nothing outside this file touches list data,
so there's no reason for database.py to know these tables exist — see
skills/README.md, "Owning your own persistence".
"""

import logging
from datetime import datetime, timezone

import database
from skills.base import (
    ClarificationNeeded,
    Skill,
    SlotSpec,
    parse_value_string,
    parse_value_string_list,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage — owned entirely by this skill module. See database.py's
# register_schema()/get_connection() docstrings for the contract.
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lists (
    id         INTEGER PRIMARY KEY,
    name       TEXT    UNIQUE NOT NULL COLLATE NOCASE,
    created_at TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS list_items (
    id         INTEGER PRIMARY KEY,
    list_id    INTEGER NOT NULL REFERENCES lists(id),
    content    TEXT    NOT NULL,
    done       INTEGER NOT NULL DEFAULT 0,
    created_at TEXT    NOT NULL
);
"""

database.register_schema(_SCHEMA)


def _get_or_create_list(name: str) -> int:
    """
    Return the id of the list with the given name, creating it if needed.
    Name comparison is case-insensitive (COLLATE NOCASE on the column).
    """
    conn = database.get_connection()
    now = datetime.now(timezone.utc).isoformat()
    # INSERT OR IGNORE leaves the row untouched if the name already exists.
    conn.execute(
        "INSERT OR IGNORE INTO lists (name, created_at) VALUES (?, ?)",
        (name, now),
    )
    conn.commit()
    row = conn.execute(
        "SELECT id FROM lists WHERE name = ?", (name,)
    ).fetchone()
    return row["id"]


def _add_list_item(list_name: str, content: str) -> None:
    """
    Add an item to the named list, creating the list if it doesn't exist.
    """
    list_id = _get_or_create_list(list_name)
    now = datetime.now(timezone.utc).isoformat()
    conn = database.get_connection()
    conn.execute(
        "INSERT INTO list_items (list_id, content, done, created_at) VALUES (?, ?, 0, ?)",
        (list_id, content, now),
    )
    conn.commit()
    log.debug("List item added: list=%r item=%r", list_name, content)


def _get_list_items(list_name: str) -> list[str]:
    """
    Return the content of all undone items in the named list, in insertion
    order. Returns an empty list if the list doesn't exist or has no items.
    """
    conn = database.get_connection()
    row = conn.execute(
        "SELECT id FROM lists WHERE name = ?", (list_name,)
    ).fetchone()

    if row is None:
        return []

    rows = conn.execute(
        "SELECT content FROM list_items WHERE list_id = ? AND done = 0 "
        "ORDER BY id ASC",
        (row["id"],),
    ).fetchall()

    return [r["content"] for r in rows]


def _clear_list(list_name: str) -> int:
    """
    Delete every item in the named list (done or not). Returns the number
    of items removed — 0 if the list doesn't exist or was already empty.
    The list row itself is left in place, empty, so it keeps existing for
    NOCASE matching and any future reads/adds.
    """
    conn = database.get_connection()
    row = conn.execute(
        "SELECT id FROM lists WHERE name = ?", (list_name,)
    ).fetchone()

    if row is None:
        return 0

    cur = conn.execute(
        "DELETE FROM list_items WHERE list_id = ?", (row["id"],)
    )
    conn.commit()
    log.debug("List cleared: list=%r items_removed=%d", list_name, cur.rowcount)
    return cur.rowcount


# ---------------------------------------------------------------------------
# Skill definitions
# ---------------------------------------------------------------------------

_LIST_NAME_DESCRIPTION = (
    "The name of the list. "
    "Return a JSON object with key: value (string). "
    "Example: 'the shopping list' → {\"value\": \"shopping\"}"
)

PROMPT_BLOCK_ADD = """\
  list_add
    list_name   (string)        name of the list (e.g. "shopping", "todo")
    list_items  (array of string) every item the user wants added. The user
                                   may mention one item or several in the
                                   same request — always return a JSON array,
                                   even for a single item.
                                   e.g. ["milk"] or ["milk", "bread", "eggs"]
"""

PROMPT_BLOCK_READ = """\
  list_read
    list_name  (string) name of the list to read out
"""

PROMPT_BLOCK_CLEAR = """\
  list_clear
    list_name  (string) name of the list to clear, empty, or delete. Use
                        this intent for phrases like "clear the shopping
                        list", "empty my todo list", or "delete the
                        groceries list" — all mean the same thing: remove
                        every item currently on that list.
"""

PROMPT_BLOCK_MERGE = """\
  list_merge
    source_list       (string) the list whose items should be moved
    destination_list  (string) the list those items should be moved into.
                                After a merge, source_list ends up empty
                                and destination_list has both lists' items.
                                e.g. "merge my groceries list into shopping"
                                → source_list="groceries",
                                  destination_list="shopping"
"""


def _join_items(items: list[str]) -> str:
    """
    Join item names into a natural spoken list:
    "milk" / "milk and eggs" / "milk, eggs, and bread"
    """
    capitalized = [item.capitalize() for item in items]
    if len(capitalized) == 1:
        return capitalized[0]
    if len(capitalized) == 2:
        return f"{capitalized[0]} and {capitalized[1]}"
    return ", ".join(capitalized[:-1]) + f", and {capitalized[-1]}"


async def handle_add(slots: dict, satellite_ip: str) -> str | None:
    list_name = slots.get("list_name", "").strip()
    raw_items = slots.get("list_items", [])
    if isinstance(raw_items, str):
        # Model returned a bare string instead of an array — treat it as a
        # single item rather than failing the whole request.
        raw_items = [raw_items]
    elif not isinstance(raw_items, list):
        raw_items = []
    items = [str(item).strip() for item in raw_items if str(item).strip()]

    if not items:
        raise ClarificationNeeded(
            intent="list_add",
            slots=slots,
            missing_slot="list_items",
            question="Please specify item.",
        )
    if not list_name:
        raise ClarificationNeeded(
            intent="list_add",
            slots=slots,
            missing_slot="list_name",
            question="Please specify list name.",
        )

    for item in items:
        _add_list_item(list_name=list_name, content=item)

    log.info("List item(s) added: list=%r items=%r", list_name, items)
    return f"{_join_items(items)} added to {list_name} list."


async def handle_read(slots: dict, satellite_ip: str) -> str | None:
    list_name = slots.get("list_name", "").strip()

    if not list_name:
        raise ClarificationNeeded(
            intent="list_read",
            slots=slots,
            missing_slot="list_name",
            question="Please specify list name.",
        )

    items = _get_list_items(list_name)

    log.info("List read: list=%r items=%d", list_name, len(items))

    if not items:
        return f"{list_name.capitalize()} list contains no items."

    # Spoken as: "Shopping list. Earl Grey. Milk. Bread."
    item_str = ". ".join(item.capitalize() for item in items)
    return f"{list_name.capitalize()} list. {item_str}."


async def handle_clear(slots: dict, satellite_ip: str) -> str | None:
    list_name = slots.get("list_name", "").strip()

    if not list_name:
        raise ClarificationNeeded(
            intent="list_clear",
            slots=slots,
            missing_slot="list_name",
            question="Which list would you like to clear?",
        )

    removed = _clear_list(list_name)

    log.info("List cleared: list=%r items_removed=%d", list_name, removed)

    if removed == 0:
        return f"{list_name.capitalize()} list is already empty."
    return f"{list_name.capitalize()} list cleared."


async def handle_merge(slots: dict, satellite_ip: str) -> str | None:
    source_list = slots.get("source_list", "").strip()
    destination_list = slots.get("destination_list", "").strip()

    if not source_list:
        raise ClarificationNeeded(
            intent="list_merge",
            slots=slots,
            missing_slot="source_list",
            question="Which list should be merged?",
        )
    if not destination_list:
        raise ClarificationNeeded(
            intent="list_merge",
            slots=slots,
            missing_slot="destination_list",
            question="Which list should it be merged into?",
        )

    # Nothing to do if someone asks to merge a list into itself — treat as
    # a no-op rather than duplicating every item.
    if source_list.lower() == destination_list.lower():
        return f"{source_list.capitalize()} is already one list."

    items = _get_list_items(source_list)

    if not items:
        log.info(
            "List merge skipped, source empty: source=%r destination=%r",
            source_list, destination_list,
        )
        return f"{source_list.capitalize()} list has no items to merge."

    for item in items:
        _add_list_item(list_name=destination_list, content=item)
    _clear_list(source_list)

    log.info(
        "List merged: source=%r destination=%r items=%d",
        source_list, destination_list, len(items),
    )
    return (
        f"Merged {len(items)} item{'s' if len(items) != 1 else ''} "
        f"from {source_list} into {destination_list} list."
    )


SKILL_LIST_ADD = Skill(
    intent="list_add",
    prompt_block=PROMPT_BLOCK_ADD,
    handler=handle_add,
    slot_specs={
        "list_name": SlotSpec(description=_LIST_NAME_DESCRIPTION, parse=parse_value_string),
        "list_items": SlotSpec(
            description=(
                "The item(s) the user wants added to the list. There may be one or "
                "several mentioned in the same reply. "
                "Return a JSON object with key: value, where value is a JSON array "
                "of strings — even for a single item. "
                'Example: "milk and a dozen eggs" → {"value": ["milk", "a dozen eggs"]}'
            ),
            parse=parse_value_string_list,
        ),
    },
)

SKILL_LIST_READ = Skill(
    intent="list_read",
    prompt_block=PROMPT_BLOCK_READ,
    handler=handle_read,
    slot_specs={
        "list_name": SlotSpec(description=_LIST_NAME_DESCRIPTION, parse=parse_value_string),
    },
)

SKILL_LIST_CLEAR = Skill(
    intent="list_clear",
    prompt_block=PROMPT_BLOCK_CLEAR,
    handler=handle_clear,
    slot_specs={
        "list_name": SlotSpec(description=_LIST_NAME_DESCRIPTION, parse=parse_value_string),
    },
)

SKILL_LIST_MERGE = Skill(
    intent="list_merge",
    prompt_block=PROMPT_BLOCK_MERGE,
    handler=handle_merge,
    slot_specs={
        "source_list": SlotSpec(
            description=(
                "The list whose items should be moved out of it. "
                "Return a JSON object with key: value (string). "
                'Example: "merge groceries into shopping" → {"value": "groceries"}'
            ),
            parse=parse_value_string,
        ),
        "destination_list": SlotSpec(
            description=(
                "The list the items should end up on. "
                "Return a JSON object with key: value (string). "
                'Example: "merge groceries into shopping" → {"value": "shopping"}'
            ),
            parse=parse_value_string,
        ),
    },
)

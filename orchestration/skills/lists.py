"""
skills/lists.py

Built-in list skills: "list_add" and "list_read". They share storage, a
"list name" concept, and a join-items helper, so they live together in one
file with two Skill objects rather than being artificially split into two
files — a skill module is the unit of *sharing*, not a hard one-file-per-
intent rule.
"""

import logging

import database
from skills.base import (
    ClarificationNeeded,
    Skill,
    SlotSpec,
    parse_value_string,
    parse_value_string_list,
)

log = logging.getLogger(__name__)

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
        database.add_list_item(list_name=list_name, content=item)

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

    items = database.get_list_items(list_name)

    log.info("List read: list=%r items=%d", list_name, len(items))

    if not items:
        return f"{list_name.capitalize()} list contains no items."

    # Spoken as: "Shopping list. Earl Grey. Milk. Bread."
    item_str = ". ".join(item.capitalize() for item in items)
    return f"{list_name.capitalize()} list. {item_str}."


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

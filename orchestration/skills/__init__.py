"""
skills/

Each module in this package defines one or more Skill objects — see
skills/base.py for the interface and skills/README.md for the full guide
to writing one.

Skills are NOT auto-discovered by dropping a file in this folder. See
skills/registry.py — that's the one place a skill gets enabled, on
purpose: a skill is arbitrary Python with full access to the database and
the satellite network, so turning one on should be a considered, auditable
action.
"""

"""
intent_router.py

Optional pre-filtering step in front of intent extraction: instead of
handing the LLM every registered skill's prompt_block on every request —
which grows the system prompt as more skills are added, and past some
point degrades a small model's extraction accuracy — embed each skill's
purpose once at startup, embed the incoming transcript, and only include
the nearest few skills' prompt_blocks in that request's system prompt.

This is a shortlist, not a classifier: it never itself decides "the intent
is X" — that's still entirely the extraction LLM's job, just working from
a smaller menu. A skill wrongly left off the shortlist is a harder failure
than the prompt-bloat problem this exists to solve, so every design choice
here errs toward including too much rather than too little:

  - SHORTLIST_SIZE is comfortably larger than however many intents you'd
    expect to be genuinely confusable with each other. A few unnecessary
    inclusions cost a few dozen tokens; a wrongly-excluded intent costs a
    wrong or "unknown" result outright.
  - Every intent in _ALWAYS_INCLUDE_INTENTS rides along regardless of
    score — "unknown" is the fallback for genuinely unmatched input, and
    it has to stay reachable even though it doesn't embed near anything.
  - If nothing scores above MIN_CONFIDENT_SIMILARITY for this transcript,
    the router doesn't trust its own shortlist and returns every skill
    instead — i.e. exactly today's un-filtered behaviour. Filtering only
    activates when the router is actually confident.
  - If Ollama is unreachable, or startup embedding failed, routing is
    disabled for the whole process and every request gets every skill —
    the same fail-open reasoning as the threshold above, just applied at
    process scope instead of per-request.

None of this touches slot-filling — clarification turns already use a
narrow, single-slot prompt (see orchestration.py's
_slot_fill_system_prompt) that was never the problem this solves.

Skill authors don't have to do anything for their skill to route
correctly: a Skill with no router_hint falls back to its intent name
(underscores turned to spaces) — a much weaker signal, but never a hard
failure. A warning is logged per skill so the gap is visible. See
skills/base.py's Skill docstring for the router_hint field.

Uses the same Ollama embedding model as skills/music.py's mood search
(nomic-embed-text) — a second embedding model loaded in Ollama for the
identical job ("is transcript X near skill-purpose Y in meaning") would
just cost more VRAM for no benefit. Deliberately does NOT use sqlite-vec
here: at the scale of a few dozen skills (as opposed to music's however-
many-thousand tracks), brute-force cosine similarity over an in-memory
list is simpler than standing up a vector index and just as fast — there
is no meaningful search-speed problem at this scale. Revisit that
decision if the skill count ever gets into the hundreds.
"""

import json
import logging
import math
import urllib.error
import urllib.request

from skills.base import Skill

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CONFIGURE: must match skills/music.py's embedding model — a mismatch
# would mean comparing vectors from two different embedding spaces, which
# produces meaningless similarity scores. See that module's OLLAMA_BASE_URL
# / EMBED_MODEL comment.
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"

# How many skills survive the shortlist, before _ALWAYS_INCLUDE_INTENTS is
# added on top. Deliberately generous — see the module docstring for why
# over-inclusion is the safe direction to err in.
SHORTLIST_SIZE = 8

# If the best-scoring skill for this transcript doesn't clear this cosine
# similarity, the router doesn't trust the shortlist and every skill is
# used instead (today's un-filtered behaviour). Cosine similarity ranges
# -1..1; 0.35 is a conservative floor. If you find routing is filtering
# out the correct intent in practice, raise this instead of lowering it —
# check the "Shortlisted N skill(s) for ..." info log to see what it
# actually included, and the "Routing scores for ..." debug log to see
# every skill's score for that transcript, before deciding whether the
# threshold or the offending skill's router_hint needs adjusting.
MIN_CONFIDENT_SIMILARITY = 0.35

# Always present in the shortlist, regardless of score.
_ALWAYS_INCLUDE_INTENTS = {"unknown"}

# ---------------------------------------------------------------------------

_skill_vectors: list[tuple[Skill, list[float]]] = []
_ready = False


def _ollama_embed(text: str) -> list[float]:
    """
    Raises urllib.error.URLError/TimeoutError on network failure, KeyError
    if the response shape is unexpected. Callers decide how to degrade.
    """
    payload = json.dumps({"model": EMBED_MODEL, "prompt": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        body = json.loads(resp.read())
    return body["embedding"]


def _router_text_for(skill: Skill) -> str:
    hint = (skill.router_hint or "").strip()
    if hint:
        return hint
    log.warning(
        "Skill %r has no router_hint — falling back to its intent name for "
        "routing, which is a much weaker similarity signal. Add router_hint "
        "to this skill's Skill(...) definition.",
        skill.intent,
    )
    return skill.intent.replace("_", " ")


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def build(skills: list[Skill]) -> bool:
    """
    Embed every skill's router text once and hold the vectors in memory for
    the process's lifetime. Call once at startup, after probe_ollama() has
    already confirmed Ollama itself is reachable.

    Returns True if routing is usable this run, False if it should be
    disabled (e.g. nomic-embed-text isn't pulled). False is not fatal —
    routing is an optimisation, not a correctness requirement, so a caller
    should log and continue with shortlist() simply always returning None
    (every skill, every request — today's behaviour) rather than treating
    this like probe_ollama()'s hard startup failure.
    """
    global _skill_vectors, _ready
    _skill_vectors = []

    try:
        for skill in skills:
            embedding = _ollama_embed(_router_text_for(skill))
            _skill_vectors.append((skill, embedding))
    except (urllib.error.URLError, TimeoutError, KeyError) as exc:
        log.error(
            "Could not build the intent router (%s) — routing disabled for "
            "this run; every request will use the full skill list. Check "
            "that %r is pulled in Ollama (ollama pull %s).",
            exc, EMBED_MODEL, EMBED_MODEL,
        )
        _skill_vectors = []
        _ready = False
        return False

    _ready = True
    log.info("Intent router ready: %d skill(s) embedded", len(_skill_vectors))
    return True


def shortlist(transcript: str, k: int = SHORTLIST_SIZE) -> list[Skill] | None:
    """
    Return the k skills whose router text is nearest to `transcript` in
    meaning, plus every _ALWAYS_INCLUDE_INTENTS skill regardless of score.

    Returns None — meaning "don't filter, use every registered skill" — in
    any of these cases, all deliberately fail-open:
      - build() was never called, or failed at startup
      - embedding this transcript fails (a mid-run Ollama hiccup)
      - the best match doesn't clear MIN_CONFIDENT_SIMILARITY, i.e. the
        router itself isn't confident enough to narrow the menu
    """
    if not _ready or not _skill_vectors:
        return None

    try:
        query_vector = _ollama_embed(transcript)
    except (urllib.error.URLError, TimeoutError, KeyError) as exc:
        log.warning(
            "Embedding transcript for routing failed (%s) — using the full "
            "skill list for this request", exc,
        )
        return None

    scored = sorted(
        ((_cosine(query_vector, vec), skill) for skill, vec in _skill_vectors),
        key=lambda pair: pair[0],
        reverse=True,
    )
    log.debug(
        "Routing scores for %r: %s",
        transcript, [(skill.intent, round(score, 3)) for score, skill in scored],
    )

    best_score = scored[0][0] if scored else -1.0
    if best_score < MIN_CONFIDENT_SIMILARITY:
        log.info(
            "Routing confidence too low for %r (best=%.3f < %.2f) — using "
            "the full skill list for this request",
            transcript, best_score, MIN_CONFIDENT_SIMILARITY,
        )
        return None

    chosen = {skill.intent: skill for _, skill in scored[:k]}
    for _, skill in scored:
        if skill.intent in _ALWAYS_INCLUDE_INTENTS:
            chosen[skill.intent] = skill

    log.info("Shortlisted %d skill(s) for %r: %s", len(chosen), transcript, sorted(chosen))
    return list(chosen.values())

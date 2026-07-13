#!/usr/bin/env python3
"""
ollama_stub.py

Minimal HTTP stub that impersonates the Ollama API for local testing.

  GET  /api/tags  — fake model list, satisfies probe_ollama()
  POST /api/chat  — cycles through canned intent JSON responses, satisfies
                     _call_ollama() / extract_intent() in orchestration.py

Response content does not depend on the request — it just cycles through
INTENT_CYCLE on every call, regardless of the transcript sent. Useful for
exercising the dispatch/handler path without a real model running.

CONFIGURE: set MODELS to match whatever INTENT_MODEL is set to in your
orchestration config — the probe checks that the name prefix is present.
CONFIGURE: edit INTENT_CYCLE to add/remove/reorder the intents returned.
"""

import hashlib
import itertools
import json
import logging
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 11434  # default Ollama port

# CONFIGURE: model names to advertise — must include your INTENT_MODEL prefix
# and the embedding model used by intent_router.py.
MODELS = [
    "qwen2.5:3b",
    "llama3.1:8b",
    "nomic-embed-text:latest",
]

# Dimensionality of stub embeddings. nomic-embed-text outputs 768 dims; this
# stub uses the same size so the router's cosine arithmetic sees realistic
# vector shapes. Change only if you switch embedding models.
EMBEDDING_DIMS = 768

# ---------------------------------------------------------------------------

def _fake_embedding(text: str) -> list[float]:
    """
    Produce a deterministic unit vector from the SHA-256 of `text`.

    The same text always gets the same vector, and different texts get
    meaningfully different vectors — enough for the router's cosine
    similarity and confidence-threshold logic to exercise all its branches
    without a real embedding model running. The vector is L2-normalised so
    cosine similarity scores stay in the expected -1..1 range.
    """
    seed = hashlib.sha256(text.encode()).digest()
    # Expand the 32-byte seed to EMBEDDING_DIMS floats by re-hashing with an
    # incrementing counter. Each round gives 32 bytes = 8 float32-range values.
    raw: list[float] = []
    counter = 0
    while len(raw) < EMBEDDING_DIMS:
        block = hashlib.sha256(seed + counter.to_bytes(2, "little")).digest()
        for i in range(0, len(block), 4):
            # Interpret 4 bytes as a signed int, scale to [-1, 1]
            val = int.from_bytes(block[i:i+4], "little", signed=True) / 2**31
            raw.append(val)
        counter += 1
    raw = raw[:EMBEDDING_DIMS]

    # L2 normalise
    magnitude = math.sqrt(sum(v * v for v in raw))
    if magnitude == 0:
        return raw  # degenerate — shouldn't happen with SHA-256 output
    return [v / magnitude for v in raw]


# CONFIGURE: intents to cycle through on each /api/chat call, one per Phase 1
# intent type plus a couple of edge cases worth exercising in the dispatcher.
INTENT_CYCLE: list[dict] = [
    {"intent": "check_timer", "slots": {}},  # no timers running yet
    {"intent": "timer", "slots": {"label": "tea", "duration_seconds": 180}},
    {"intent": "list_add", "slots": {"list_name": "shopping", "list_items": "earl grey"}},
    {"intent": "list_read", "slots": {"list_name": "shopping"}},
    {"intent": "timer", "slots": {"label": "coffee", "hours": 0, "minutes": 1, "seconds": 30}},
    {"intent": "timer", "slots": {"label": "bag"}},
    {"hours": 0, "minutes": 3, "seconds": 30},
    {"intent": "list_add", "slots": {"list_items": ["apples"]}},
    {"value": "shopping"},
    {"intent": "list_add", "slots": {"list_name": "shopping", "list_items": ["cheese","milk","eggs"]}},
    {"intent": "check_timer", "slots": {}},  # pending timers, unlabelled query
    {"intent": "check_timer", "slots": {"label": "coffee"}},  # pending, labelled query
    {"intent": "check_timer", "slots": {"label": "pasta"}},
    {"intent": "unknown", "slots": {}},
    # Edge case: duration of zero should hit handle_timer's early-return path
    {"intent": "timer", "slots": {"label": "broken", "duration_seconds": 0}},
    {"hours": 0, "minutes": 3, "seconds": 30},
    # Edge case: missing "intent" key entirely — exercises extract_intent()'s
    # KeyError branch in orchestration.py, which should fall back to unknown.
    {"slots": {}},
]

# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)

TAGS_RESPONSE = json.dumps({
    "models": [{"name": m} for m in MODELS]
}).encode()

# itertools.cycle is not thread-safe against concurrent next() calls, and
# ThreadingHTTPServer spins up a thread per request — guard with a lock.
_cycle_lock = threading.Lock()
_intent_cycle = itertools.cycle(INTENT_CYCLE)


def _next_intent() -> dict:
    with _cycle_lock:
        return next(_intent_cycle)


class OllamaStubHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/api/tags":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(TAGS_RESPONSE)))
            self.end_headers()
            self.wfile.write(TAGS_RESPONSE)
            log.info("GET /api/tags — responded with %d model(s)", len(MODELS))
        else:
            self.send_response(404)
            self.end_headers()
            log.warning("GET %s — 404 (not stubbed)", self.path)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request_body = self.rfile.read(length)

        if self.path == "/api/chat":
            intent = _next_intent()
            # Mirrors the real Ollama /api/chat response shape: the model's
            # reply lives at body["message"]["content"] as a JSON *string*,
            # matching what _call_ollama() expects to json.loads() twice.
            response_body = json.dumps({
                "message": {"role": "assistant", "content": json.dumps(intent)}
            }).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            log.info("POST /api/chat — returned intent=%r", intent.get("intent", "<missing>"))

        elif self.path == "/api/embeddings":
            # intent_router.py calls _ollama_embed(text) which POSTs here with
            # {"model": "nomic-embed-text", "prompt": "<text>"} and expects
            # {"embedding": [float, ...]} back. The fake vector is deterministic
            # per input text so repeated calls for the same skill hint or
            # transcript always produce identical vectors — matching what a real
            # model would do for a warm cache.
            try:
                body = json.loads(request_body)
                prompt = body.get("prompt", "")
            except (json.JSONDecodeError, KeyError):
                prompt = ""

            vector = _fake_embedding(prompt)
            response_body = json.dumps({"embedding": vector}).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            log.info(
                "POST /api/embeddings — embedded %r → %d-dim vector",
                prompt[:60] + ("…" if len(prompt) > 60 else ""), EMBEDDING_DIMS,
            )

        else:
            self.send_response(404)
            self.end_headers()
            log.warning("POST %s — 404 (not stubbed)", self.path)

    # Suppress the default per-request stdout noise from BaseHTTPRequestHandler
    def log_message(self, fmt, *args):
        pass


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    server = ThreadingHTTPServer((HOST, PORT), OllamaStubHandler)
    log.info("Ollama stub listening on http://%s:%d", HOST, PORT)
    log.info("Advertising models: %s", MODELS)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Interrupted — shutting down")

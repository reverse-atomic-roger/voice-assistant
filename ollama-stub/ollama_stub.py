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

import itertools
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOST = "127.0.0.1"
PORT = 11434  # default Ollama port

# CONFIGURE: model names to advertise — must include your INTENT_MODEL prefix
MODELS = [
    "qwen2.5:3b",
    "llama3.1:8b",
]

# CONFIGURE: intents to cycle through on each /api/chat call, one per Phase 1
# intent type plus a couple of edge cases worth exercising in the dispatcher.
INTENT_CYCLE: list[dict] = [
    {"intent": "timer", "slots": {"label": "tea", "duration_seconds": 180}},
    {"intent": "list_add", "slots": {"list_name": "shopping", "item": "earl grey"}},
    {"intent": "list_add", "slots": {"list_name": "shopping", "item": "apples"}},
    {"intent": "list_read", "slots": {"list_name": "shopping"}},
    #{"intent": "converse", "slots": {}},
    {"intent": "unknown", "slots": {}},
    # Edge case: duration of zero should hit handle_timer's early-return path
    {"intent": "timer", "slots": {"label": "broken", "duration_seconds": 0}},
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
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            request_body = self.rfile.read(length)  # drained but content ignored — see module docstring

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
            log.info("POST /api/chat — returned intent=%r", intent["intent"])
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

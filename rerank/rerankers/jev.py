"""Jev as a reranker: the same two typed questions Laya gets, over HTTP.

The transport is written out here rather than imported from a client library, and
it keeps the two properties that matter for a benchmark:

  * rate limits (429) and overload (529) are retried with exponential backoff;
  * the reported latency covers only the attempt that succeeded, so waiting out
    a queue is never charged to the model's speed.

Two routes to the same model: OpenRouter by default, or TypeSafe directly when
TYPESAFE_API_KEY is set. Both pin a build, because a benchmark that cannot say
which build produced a number is not a benchmark.

A call that never succeeds is recorded as failed with `score: None`. Nothing is
invented for it: `rank_by_score` leaves that passage exactly where BM25 put it.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .laya import NOUL_QUESTION, SCORE_QUESTION, build_state

# Decision models answer at `systemone`, not `chat/completions`: the typed
# question itself goes on the wire, so there is no prompt and nothing to parse.
OPENROUTER_URL = "https://openrouter.ai/api/v1/systemone"
OPENROUTER_MODEL = "typesafe/jev-1.13-20260917"  # dated: the minor floats
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-1.13.0"  # pinned so benchmark runs are reproducible
ROOT_ENV = Path(__file__).resolve().parents[2] / ".env"  # this repo's root

# Jev accepts `choice`, `score` and `boolean`. `boolean` is the same question
# Laya calls `noul` -- no criteria, one probability back -- under a different
# name, and it answers with `probability` instead of `noul`; a body with
# `"type": "noul"` comes back 400. Only the dialect is translated: the
# instructions and the state are word for word what Laya is handed, so the two
# models are asked the identical thing.
BOOLEAN_QUESTION = {"relevance": {**NOUL_QUESTION["relevance"], "type": "boolean"}}

VARIANTS = {"jev-score": (SCORE_QUESTION, "score"), "jev-noul": (BOOLEAN_QUESTION, "probability")}


def _http_post(url, headers, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


class JevReranker:
    RETRYABLE = {429, 529}  # rate limited / overloaded

    def __init__(self, name, questions, field, api_key, provider="openrouter", post=_http_post,
                 max_retries=5, backoff_s=1.0):
        if not api_key:
            raise ValueError(
                "no Jev API key: set OPENROUTER_API_KEY (or TYPESAFE_API_KEY) in the "
                "environment or in this repo's root .env")
        self.name, self.questions, self.field = name, questions, field
        self.provider, self.post, self.max_retries, self.backoff_s = provider, post, max_retries, backoff_s
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        self.url = TYPESAFE_URL if provider == "typesafe" else OPENROUTER_URL
        self.model = TYPESAFE_MODEL if provider == "typesafe" else OPENROUTER_MODEL

    def _body(self, state):
        return {"model": self.model, "state": state, "questions": self.questions}

    def score(self, query, passage):
        state = build_state(query, passage)
        request = self._body(state)
        retries, error = 0, None
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self.post(self.url, self.headers, request)
            except urllib.error.HTTPError as e:
                error, retries = f"HTTP {e.code}: {e.reason}", attempt
                if e.code not in self.RETRYABLE or attempt == self.max_retries:
                    break
                time.sleep(self.backoff_s * 2 ** attempt)  # deliberately outside the timer
                continue
            except Exception as e:  # a connection reset should not end the run either
                error, retries = f"{type(e).__name__}: {e}", attempt
                if attempt == self.max_retries:
                    break
                time.sleep(self.backoff_s * 2 ** attempt)
                continue
            return {"score": float(response["answers"]["relevance"][self.field]),
                    "request": request, "response": response,
                    "latency_ms": (time.perf_counter() - started) * 1000,
                    "retries": attempt, "failed": False}
        return {"score": None, "request": request, "response": {"error": error},
                "latency_ms": 0.0, "retries": retries, "failed": True}


def _key(var):
    """Reads a key from the environment, then from the repo-root .env."""
    if key := os.environ.get(var):
        return key
    lines = ROOT_ENV.read_text().splitlines() if ROOT_ENV.exists() else []
    return next((l.split("=", 1)[1].strip() for l in lines if l.startswith(f"{var}=")), None)


def load(name, api_key=None, provider=None, **kwargs):
    questions, field = VARIANTS[name]
    if api_key is None and provider is None:
        # TypeSafe first when it is set: it pins the exact build. Otherwise
        # OpenRouter, which is what every other experiment in this repo uses.
        if key := _key("TYPESAFE_API_KEY"):
            api_key, provider = key, "typesafe"
        else:
            api_key, provider = _key("OPENROUTER_API_KEY"), "openrouter"
    return JevReranker(name, questions, field, api_key=api_key,
                       provider=provider or "openrouter", **kwargs)

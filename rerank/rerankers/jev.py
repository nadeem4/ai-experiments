"""Jev as a reranker: the same two typed questions Laya gets, over HTTP.

The transport is copied from `arena/arena/agents/jev.py` rather than imported --
each demo in this repo stands alone -- and it keeps that module's two properties
that matter for a benchmark:

  * rate limits (429) and overload (529) are retried with exponential backoff,
    because the arena measured 1,231 retries across 1,423 calls;
  * the reported latency covers only the attempt that succeeded, so waiting out
    a queue is never charged to the model's speed.

Two routes to the same model, same as arena: TypeSafe directly when
TYPESAFE_API_KEY is set (pinned version), otherwise the Vercel AI Gateway.

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

GATEWAY_URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"
TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-1.13.0"  # pinned so benchmark runs are reproducible
ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"  # repo root, shared by all demos

# Jev's v4 schema accepts `choice`, `score` and `boolean`. `boolean` is the same
# question Laya calls `noul` -- no criteria, one probability back -- under a
# different name, and it answers with `probability` instead of `noul`; a body
# with `"type": "noul"` comes back 400. Only the dialect is translated: the
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

    def __init__(self, name, questions, field, api_key, provider="gateway", post=_http_post,
                 max_retries=5, backoff_s=1.0):
        if not api_key:
            raise ValueError("no Jev API key: set TYPESAFE_API_KEY or AI_GATEWAY_API_KEY")
        self.name, self.questions, self.field = name, questions, field
        self.provider, self.post, self.max_retries, self.backoff_s = provider, post, max_retries, backoff_s
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if provider == "gateway":
            self.url = GATEWAY_URL
            self.headers.update({
                "ai-model-id": "typesafe-ai/jev",
                "ai-evaluation-model-specification-version": "4",
                "ai-gateway-protocol-version": "0.0.1",
                "ai-gateway-auth-method": "api-key",
            })
        else:
            self.url = TYPESAFE_URL

    def _body(self, state):
        body = {"state": state, "questions": self.questions}
        return body if self.provider == "gateway" else {"model": TYPESAFE_MODEL, **body}

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
        if key := _key("TYPESAFE_API_KEY"):
            api_key, provider = key, "typesafe"
        else:
            api_key, provider = _key("AI_GATEWAY_API_KEY"), "gateway"
    return JevReranker(name, questions, field, api_key=api_key, provider=provider or "gateway", **kwargs)

"""The wire to the Vercel AI Gateway: chat completions for the LLMs, the v4
evaluation-model route for Jev.

Written out here rather than imported from a client library -- the same choice
`rerank/rerankers/jev.py` makes, and for the same reason: a benchmark has to be
able to say exactly what went over the wire and which clock each number came off.

Two properties matter.

  * Rate limits and transport errors are retried with exponential backoff, and the
    sleep is deliberately outside the timer. `latency_ms` covers only the attempt
    that succeeded, so queueing is never charged to a model's speed; `wall_ms`
    covers the whole call including backoff, so it never disappears either.
  * An answer is never asked for twice. One call, one answer. A completion that
    cannot be parsed is a result -- see `parse.py` -- not a reason to call again.
"""
import json
import time
import urllib.error
import urllib.request

from . import prompts

CHAT_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
EVALUATION_URL = "https://ai-gateway.vercel.sh/v4/ai/evaluation-model"

# One key for both tasks, so no model gets a differently named question.
QUESTION_ID = "decision"
MAX_TOKENS = 64  # an option name and the JSON around it; 5 and 77 options alike

# 429 rate limited, 529 overloaded, 5xx transport. Retrying these is not
# retry-until-valid: none of them is an answer.
RETRYABLE = {429, 500, 502, 503, 504, 529}


def _http_post(url, headers, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as r:
        return json.load(r)


class _Model:
    """The retry loop and the two clocks, shared by both routes."""

    def __init__(self, name, model_id, api_key, post=_http_post, max_retries=5, backoff_s=1.0,
                 sleep=time.sleep):
        if not api_key:
            raise ValueError("no gateway key: set AI_GATEWAY_API_KEY")
        self.name, self.model_id = name, model_id
        self.post, self.max_retries, self.backoff_s, self.sleep = post, max_retries, backoff_s, sleep
        self.headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def _call(self, url, body):
        wall_started = time.perf_counter()
        error, retries = None, 0
        for attempt in range(self.max_retries + 1):
            started = time.perf_counter()
            try:
                response = self.post(url, self.headers, body)
            except urllib.error.HTTPError as e:
                error, retries = f"HTTP {e.code}: {e.reason}", attempt
                if e.code not in RETRYABLE or attempt == self.max_retries:
                    break
                self.sleep(self.backoff_s * 2 ** attempt)  # deliberately outside the timer
                continue
            except Exception as e:  # a connection reset should not end the run either
                error, retries = f"{type(e).__name__}: {e}", attempt
                if attempt == self.max_retries:
                    break
                self.sleep(self.backoff_s * 2 ** attempt)
                continue
            return {"response": response, "latency_ms": (time.perf_counter() - started) * 1000,
                    "wall_ms": (time.perf_counter() - wall_started) * 1000,
                    "retries": attempt, "failed": False, "error": None}
        return {"response": None, "latency_ms": None,
                "wall_ms": (time.perf_counter() - wall_started) * 1000,
                "retries": retries, "failed": True, "error": error}


class ChatModel(_Model):
    """An LLM, asked in the gateway's structured-output mode where it exposes one.

    Structured mode is rule one of the fairness rules -- give the LLMs their best
    shot -- and `structured` records which arm the call ran in, because it changes
    both latency and validity."""

    def __init__(self, *args, temperature=0, structured=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature, self.structured = temperature, structured

    def decide(self, state, instructions, criteria):
        body = {
            "model": self.model_id,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompts.render(instructions, state, criteria)}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.structured:
            body["response_format"] = prompts.schema(criteria)
        out = self._call(CHAT_URL, body)
        response = out["response"] or {}
        usage = response.get("usage") or {}
        choices = response.get("choices") or [{}]
        message = choices[0].get("message") or {}
        return {
            **out,
            "request": body,
            "content": message.get("content") if out["response"] else None,
            "finish_reason": choices[0].get("finish_reason"),
            "structured": self.structured,
            "temperature": self.temperature,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "market_cost": _market_cost(usage.get("market_cost"), response),
            "usage": usage,
        }


class JevModel(_Model):
    """Jev over the v4 evaluation-model route: the typed question itself goes on the
    wire, so there is no prompt and nothing to parse back."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers.update({
            "ai-model-id": self.model_id,
            "ai-evaluation-model-specification-version": "4",
            "ai-gateway-protocol-version": "0.0.1",
            "ai-gateway-auth-method": "api-key",
        })

    def decide(self, state, instructions, criteria):
        body = {"state": state, "questions": {
            QUESTION_ID: {"type": "choice", "instructions": instructions, "criteria": criteria}}}
        out = self._call(EVALUATION_URL, body)
        response = out["response"] or {}
        answer = (response.get("answers") or {}).get(QUESTION_ID) or {}
        usage = response.get("usage") or {}
        return {
            **out,
            "request": body,
            "content": None,
            "choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "structured": "typed",
            "temperature": None,  # no sampling: one forward pass, one distribution
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "market_cost": _market_cost(None, response),
            "usage": usage,
        }


def _market_cost(from_usage, response):
    """The gateway's own per-call figure. Cost is never computed from a token count
    and a rate off a pricing page -- only this number is reported."""
    if from_usage is not None:
        return float(from_usage)
    gateway = ((response or {}).get("providerMetadata") or {}).get("gateway") or {}
    cost = gateway.get("marketCost")
    return float(cost) if cost is not None else None

"""The wire to OpenRouter: chat completions for the LLMs, the `systemone` route
for Jev.

Written out here rather than imported from a client library -- the same choice
`rerank/rerankers/jev.py` makes, and for the same reason: a benchmark has to be
able to say exactly what went over the wire and which clock each number came off.

Three properties matter.

  * Rate limits and transport errors are retried with exponential backoff, and the
    sleep is deliberately outside the timer. `latency_ms` covers only the attempt
    that succeeded, so queueing is never charged to a model's speed; `wall_ms`
    covers the whole call including backoff, so it never disappears either.
  * An answer is never asked for twice. One call, one answer. A completion that
    cannot be parsed is a result -- see `parse.py` -- not a reason to call again.
  * Every record carries `transport`. The experiment previously ran over the
    Vercel AI Gateway, and those numbers are not comparable to these: a different
    provider, a different route and a different set of models. Tagging every call
    is what makes averaging the two together impossible rather than merely
    discouraged.

Two route facts, established by probing the live API rather than assumed:

  * **Jev is called at `POST /api/v1/systemone`**, not `chat/completions`, which
    rejects the decision shape with `Input required: specify "prompt" or
    "messages"`. The body is `{"model", "state", "questions"}` -- the model id goes
    in the body, where the Vercel route took it as an `ai-model-id` header -- and
    the response carries `answers` plus a `usage` block with `input_tokens`,
    `output_tokens` and `cost` in dollars.
  * The LLMs use the ordinary `POST /api/v1/chat/completions` with a bearer token,
    keeping structured output, `strict`, the `enum` of option keys and temperature
    zero exactly as the gateway arm did. Their `usage` block reports `cost` too, so
    no cost anywhere in this experiment is computed from a pricing page.
"""
import json
import time
import urllib.error
import urllib.request

from . import prompts
from .catalog import JEV_ID  # re-exported: the pinned, dated Jev build

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEMONE_URL = "https://openrouter.ai/api/v1/systemone"
KEY_URL = "https://openrouter.ai/api/v1/key"

TRANSPORT = "openrouter"

# One key for both tasks, so no model gets a differently named question.
QUESTION_ID = "decision"

# Headroom, not a budget. An option name and the JSON around it is under 20 tokens,
# but the current tiers are reasoning models: one of them spent 57 thinking tokens
# and returned no answer at all inside the old 64-token ceiling, which would have
# been recorded as a validity failure that was really an instrument failure.
MAX_TOKENS = 512

# 429 rate limited, 529 overloaded, 5xx transport. Retrying these is not
# retry-until-valid: none of them is an answer.
RETRYABLE = {429, 500, 502, 503, 504, 529}


def _http_post(url, headers, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(request, timeout=120) as r:
        return json.load(r)


def balance(api_key):
    """What is left of the account's monthly cap, for the pre-run check.

    Returns the raw `/key` block; `limit` is the cap and `limit_remaining` what is
    left of it. Never used to compute a reported cost -- only to decide whether a
    run should start."""
    request = urllib.request.Request(KEY_URL, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(request, timeout=60) as r:
        return json.load(r)["data"]


class _Model:
    """The retry loop and the two clocks, shared by both routes."""

    def __init__(self, name, model_id, api_key, post=_http_post, max_retries=5, backoff_s=1.0,
                 sleep=time.sleep):
        if not api_key:
            raise ValueError("no OpenRouter key: set OPENROUTER_API_KEY")
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
                    "retries": attempt, "failed": False, "error": None,
                    "transport": TRANSPORT}
        return {"response": None, "latency_ms": None,
                "wall_ms": (time.perf_counter() - wall_started) * 1000,
                "retries": retries, "failed": True, "error": error,
                "transport": TRANSPORT}


class ChatModel(_Model):
    """An LLM, asked in OpenRouter's structured-output mode.

    Structured mode is rule one of the fairness rules -- give the LLMs their best
    shot -- and `structured` records which arm the call ran in, because it changes
    both latency and validity.

    `temperature` and `reasoning_effort` follow the same rule as each other: sent
    only where the catalog says the model accepts the parameter, recorded either
    way, never silently assumed. The current GPT and Claude tiers take neither a
    temperature nor an essay."""

    def __init__(self, *args, temperature=0, structured=True, reasoning_effort=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature, self.structured = temperature, structured
        self.reasoning_effort = reasoning_effort

    def decide(self, state, instructions, criteria):
        body = {
            "model": self.model_id,
            "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompts.render(instructions, state, criteria)}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
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
            "reasoning_effort": self.reasoning_effort,
            "input_tokens": usage.get("prompt_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
            "cost": _cost(usage),
            "usage": usage,
        }


class JevModel(_Model):
    """Jev over the `systemone` route: the typed question itself goes on the wire,
    so there is no prompt and nothing to parse back."""

    def decide(self, state, instructions, criteria):
        body = {"model": self.model_id, "state": state, "questions": {
            QUESTION_ID: {"type": "choice", "instructions": instructions, "criteria": criteria}}}
        out = self._call(SYSTEMONE_URL, body)
        response = out["response"] or {}
        answer = (response.get("answers") or {}).get(QUESTION_ID) or {}
        usage = response.get("usage") or {}
        return {
            **out,
            "request": body,
            "content": None,
            "choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "confidence": answer.get("confidence"),
            "structured": "typed",
            "temperature": None,  # no sampling: one forward pass, one distribution
            "reasoning_effort": None,
            "input_tokens": usage.get("input_tokens"),
            "output_tokens": usage.get("output_tokens"),
            "reasoning_tokens": None,
            "cost": _cost(usage),
            "usage": usage,
        }


def _cost(usage):
    """OpenRouter's own per-call figure, in dollars. Cost is never computed from a
    token count and a rate off a pricing page -- only this number is reported."""
    cost = (usage or {}).get("cost")
    return float(cost) if cost is not None else None

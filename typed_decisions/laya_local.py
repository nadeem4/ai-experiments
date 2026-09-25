"""Laya, the open-weights decision model, running on this machine.

It is in its own group in every table and labelled everywhere, because its number
is not comparable to a hosted call: **there is no usable GPU here, so this is CPU
latency** with no network in it at all. It is included because it is the only arm
that shows what a typed decision costs when nobody is charging for it.

Laya reports no token counts, so `input_tokens` and `output_tokens` are None rather
than estimated, and there is no per-call price, so `cost` is 0.

Its `transport` is `local-cpu`, which is what keeps it out of every hosted table:
the transport is part of the report's grouping key, so a local CPU number can
never be averaged with a network round trip.
"""
import os
import time

TRANSPORT = "local-cpu"
WEIGHTS = os.environ.get("LAYA_PATH", r"C:\projects\jev_demo\arena\models\laya")
QUESTION_ID = "decision"


class LayaModel:
    def __init__(self, name="laya", weights=WEIGHTS, agent=None):
        self.name, self.model_id, self.weights = name, f"laya ({weights})", weights
        self._agent = agent

    @property
    def agent(self):
        if self._agent is None:
            import laya

            started = time.perf_counter()
            self._agent = laya.load(self.weights)
            print(f"  loaded laya from {self.weights} in {time.perf_counter() - started:.1f}s (cpu)")
        return self._agent

    def config(self):
        return {k: self.agent.cfg.get(k) for k in ("head_max_len", "max_len")}

    def decide(self, state, instructions, criteria):
        questions = {QUESTION_ID: {"type": "choice", "instructions": instructions, "criteria": criteria}}
        started = time.perf_counter()
        try:
            response = self.agent.predict(state, questions)
        except Exception as e:
            elapsed = (time.perf_counter() - started) * 1000
            return {"request": {"state": state, "questions": questions}, "response": None,
                    "content": None, "choice": None, "probabilities": None, "structured": "typed",
                    "temperature": None, "latency_ms": None, "wall_ms": elapsed, "retries": 0,
                    "failed": True, "error": f"{type(e).__name__}: {e}",
                    "input_tokens": None, "output_tokens": None, "cost": 0.0, "usage": {}, "transport": TRANSPORT,
                    "reasoning_effort": None, "reasoning_tokens": None}
        elapsed = (time.perf_counter() - started) * 1000
        answer = response["answers"][QUESTION_ID]
        return {
            "request": {"state": state, "questions": questions},
            "response": response,
            "content": None,
            "choice": answer.get("choice"),
            "probabilities": answer.get("probabilities"),
            "structured": "typed",
            "temperature": None,  # one forward pass, no sampling
            "latency_ms": elapsed,
            "wall_ms": elapsed,  # no network, so nothing to separate
            "retries": 0,
            "failed": False,
            "error": None,
            "input_tokens": None,  # laya reports none; not estimated
            "output_tokens": None,
            "cost": 0.0,  # local: no per-call price
            "transport": TRANSPORT,
            "reasoning_effort": None,
            "reasoning_tokens": None,
            "usage": {},
        }

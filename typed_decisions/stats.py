"""Percentiles, a bootstrap CI, agreement, and the estimate that guards a run.

Plain Python and numpy. Percentiles are nearest-rank rather than interpolated, so
every latency reported is a latency some call actually had.
"""
import math

import numpy as np


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q / 100 * len(ordered)))
    return ordered[rank - 1]


def bootstrap_ci(values, n=1000, seed=0, alpha=0.05):
    """Percentile CI of the mean over `n` resamples with replacement."""
    if not values:
        return (None, None)
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def agreement(a, b):
    """-> (share, n) over the examples both models answered.

    Used where there is no ground truth. An unusable answer -- None -- agrees with
    nothing, including another None: two models that both failed have not agreed on
    anything, they have both failed."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return (None, 0)
    same = sum(1 for k in shared if a[k] is not None and a[k] == b[k])
    return (same / len(shared), len(shared))


def cost_per_correct_decision(cost_per_call, accuracy):
    """The headline number: what one *usable* decision costs.

    Dollars per call is misleading whenever accuracy differs between arms -- a
    model at half the price and half the accuracy costs the same per decision you
    can actually act on. Undefined rather than infinite at zero accuracy: a model
    that is never right has no cost per correct decision, and `inf` would sort
    into a table as though somebody had measured it."""
    if cost_per_call is None or not accuracy:
        return None
    return cost_per_call / accuracy


def tail_ratio(p95, p50):
    """p95 / p50. A model with a 10x tail is worse in production than a slower
    steady one, and a p50-only table hides that completely: the earlier pilot had
    one model at 689 ms p50 against 7,093 ms p95."""
    if p95 is None or not p50:
        return None
    return p95 / p50


def disagreement_rate(first, second):
    """-> (share, n) over the examples that were actually answered twice.

    Determinism: a decision model should be exactly reproducible, an LLM at
    temperature 0 often is not. The denominator is the repeated subset, not the
    whole run, and `None` -- the absence of an answer -- is never evidence of
    reproducibility, so two failures count as a disagreement rather than as a
    model agreeing with itself."""
    shared = sorted(set(first) & set(second))
    if not shared:
        return (None, 0)
    differ = sum(1 for k in shared
                 if first[k] is None or second[k] is None or first[k] != second[k])
    return (differ / len(shared), len(shared))


def call_cost(input_tokens, output_tokens, pricing):
    """Only for the pre-run estimate. Every cost that gets reported comes off the
    provider's own per-call `cost`, never off a rate and a token count."""
    return input_tokens * float(pricing["input"]) + output_tokens * float(pricing["output"])


def estimate(plan):
    by_model = {p["model"]: (p["calls"] + p["warmup"]) * p["cost_per_call"] for p in plan}
    return {"by_model": by_model, "total_usd": sum(by_model.values())}


def needs_confirmation(total_usd, threshold):
    return total_usd > threshold

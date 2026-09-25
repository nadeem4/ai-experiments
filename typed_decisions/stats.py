"""Percentiles, a bootstrap CI, the position-bias measures, and the run's spend estimate.

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


# --- position bias -----------------------------------------------------------
#
# Two arms, because they answer different questions. The flip rate says how much
# a model's answer moves when nothing but the option order changes -- every
# movement is pure error, since the example and the options are identical. The
# controlled gold placement says *which* positions it favours, which the flip
# rate structurally cannot tell you: a model that flips on every example might be
# noisy, or might be reliably picking whatever is at the top.


def flip_rate(answers_by_example, n_orders=3):
    """-> {"rate", "n", "n_incomplete"} over examples answered under every order.

    An example flips if its answers are not all the same string. `None` -- an
    unusable answer -- is never evidence of stability, so three failures count as
    a flip rather than as a model agreeing with itself.

    An example missing one of the orders is **excluded and counted**, not scored:
    a model that failed a call has demonstrated neither stability nor a flip, and
    letting it count either way would quietly move the number."""
    complete, incomplete = [], 0
    for answers in answers_by_example.values():
        if len(answers) != n_orders:
            incomplete += 1
            continue
        complete.append(answers)
    if not complete:
        return {"rate": None, "n": 0, "n_incomplete": incomplete}
    flipped = sum(1 for a in complete if None in a or len(set(a)) > 1)
    return {"rate": flipped / len(complete), "n": len(complete), "n_incomplete": incomplete}


def accuracy_by_position(correct_by_placement):
    """-> {"accuracy": {placement: acc}, "spread": float, "n": {placement: n}}.

    `spread` is max minus min across the three placements and is the causal
    measure the experiment exists for. It is `None` unless all three placements
    were actually measured: two out of three is not a spread, and reporting one
    would be a comparison the run never made."""
    accuracy, counts = {}, {}
    for placement, values in correct_by_placement.items():
        scored = [v for v in values if v is not None]
        counts[placement] = len(scored)
        accuracy[placement] = (sum(scored) / len(scored)) if scored else None
    complete = [accuracy.get(p) for p in ("first", "middle", "last")]
    spread = (max(complete) - min(complete)) if all(v is not None for v in complete) else None
    return {"accuracy": accuracy, "spread": spread, "n": counts}


def chosen_position(choice, options_shown):
    """Where the answer sat in the list this call actually showed, 0-based.

    Read off the order that went over the wire rather than off the canonical
    order, because the point of the arm is that they differ."""
    if choice is None:
        return None
    try:
        return list(options_shown).index(choice)
    except ValueError:
        return None


def position_histogram(positions, n_options, n_bins=None):
    """-> {"shares", "edges", "mean_normalised", "n"}.

    What *kind* of positional bias a model has, rather than only how much. With
    four options every option gets its own bin; with 151 the positions are binned
    into equal-width slices so the shape is readable at all.

    `mean_normalised` is 0 when the model always picks the first option shown and
    1 when it always picks the last, so one number says which end it leans to."""
    positions = [p for p in positions if p is not None]
    if not positions or n_options < 1:
        return {"shares": None, "edges": None, "mean_normalised": None, "n": 0}
    bins = min(n_options, n_bins or n_options)
    counts = [0] * bins
    for p in positions:
        counts[min(bins - 1, p * bins // n_options)] += 1
    edges = [i * n_options / bins for i in range(bins + 1)]
    denominator = (n_options - 1) or 1
    return {
        "shares": [c / len(positions) for c in counts],
        "edges": edges,
        "mean_normalised": sum(positions) / len(positions) / denominator,
        "n": len(positions),
    }


# --- paired comparison -------------------------------------------------------


def paired(a, b, n=1000, seed=0, alpha=0.05):
    """-> {"n", "mean_diff", "ci"} over the examples **both** models answered.

    Per-item differences, not two overlapping intervals: shared per-item
    variation -- some examples are simply harder -- swamps the effect otherwise,
    and two intervals that overlap say nothing about how the two compare. `n` is
    reported everywhere because the intersection can be much smaller than either
    model's own sample."""
    shared = sorted(set(a) & set(b))
    if not shared:
        return {"n": 0, "mean_diff": None, "ci": (None, None)}
    diffs = [a[k] - b[k] for k in shared]
    return {"n": len(shared), "mean_diff": sum(diffs) / len(diffs),
            "ci": bootstrap_ci(diffs, n=n, seed=seed, alpha=alpha)}

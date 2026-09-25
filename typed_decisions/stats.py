"""Every number this experiment reports, as pure functions over plain lists.

Percentiles, a bootstrap CI, paired differences, the position-bias measures,
calibration, and the estimate that guards a paid run. Percentiles are
nearest-rank rather than interpolated, so every latency reported is a latency
some call actually had.

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
    """-> {"rate", "n", "n_incomplete", "n_unusable"}.

    An example flips if its answers are not all the same. Two kinds of example
    are **excluded and counted** rather than scored, because in neither case did
    the option order cause anything:

      * *incomplete* -- the model has no record under one of the orders, so it
        demonstrated neither stability nor a flip;
      * *unusable* -- the model returned nothing usable under **any** order. That
        is a failure, and counting it as position bias would put a number in the
        bias table that the order did not cause. Laya on a 151-option list does
        exactly this: every call fails on its option budget, and a 100% flip rate
        would be a lie about why.

    An example answered under two orders and failed under the third **is** a
    flip: the answer did change with the order."""
    complete, incomplete, unusable = [], 0, 0
    for answers in answers_by_example.values():
        if len(answers) != n_orders:
            incomplete += 1
            continue
        if all(a is None for a in answers):
            unusable += 1
            continue
        complete.append(answers)
    if not complete:
        return {"rate": None, "n": 0, "n_incomplete": incomplete, "n_unusable": unusable}
    flipped = sum(1 for a in complete if len(set(a)) > 1)
    return {"rate": flipped / len(complete), "n": len(complete),
            "n_incomplete": incomplete, "n_unusable": unusable}


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


# --- calibration -------------------------------------------------------------
#
# Is the number a model hands back a probability you could act on? Only the
# decision models answer that at all: Jev and Laya return a distribution over every
# option, an LLM in structured mode returns a label. Without a probability there is
# no threshold, so "route this one to a human" is not available from that model at
# any price -- which is why the capability is a column in the report and
# calibration is present for some rows and simply absent for others.
#
# The temperature is fitted on a validation split carved out of TRAIN, never on
# test. `calibrate()` takes the two sets separately for exactly that reason.


N_BINS = 15

# The temperature is searched inside a range, not off the end of one. A fitted
# 0.001 is the optimiser running out of grid, not a calibration.
T_MIN, T_MAX = 0.25, 10.0
_GRID = [T_MIN + 0.02 * i for i in range(int((T_MAX - T_MIN) / 0.02) + 1)]

# Below this many validation rows, no temperature is reported at all. With a
# handful of confident, correct rows the likelihood is minimised by driving T
# towards zero: the fit returns the smallest value on the grid and a scaled ECE
# of exactly zero, which is an artefact of the sample size and would be read as a
# calibration result. Refusing to fit is the honest outcome.
MIN_VALIDATION = 30


def ece(confidences, correct, n_bins=N_BINS):
    """Expected calibration error: sum over bins of |accuracy - confidence|,
    weighted by the share of answers in the bin."""
    if len(confidences) != len(correct):
        raise ValueError(f"{len(confidences)} confidences against {len(correct)} outcomes")
    if not confidences:
        return None
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, ok))
    total = len(confidences)
    return sum(
        len(rows) / total * abs(sum(ok for _, ok in rows) / len(rows)
                                - sum(c for c, _ in rows) / len(rows))
        for rows in bins if rows
    )


def _log(p):
    """log with a floor, because a model that returns a hard 0 for an option is
    common and -inf would end the fit rather than inform it."""
    return math.log(max(float(p), 1e-12))


def _nll(rows, temperature):
    total = 0.0
    for probs, gold in rows:
        scaled = {k: _log(v) / temperature for k, v in probs.items()}
        peak = max(scaled.values())
        denominator = sum(math.exp(v - peak) for v in scaled.values())
        total -= (scaled.get(gold, _log(0) / temperature) - peak) - math.log(denominator)
    return total / len(rows)


def fit_temperature(rows):
    """-> the scalar T minimising negative log likelihood on `rows`.

    `rows` is [({option: probability}, gold option)] and must come from the
    **validation** split. A coarse grid then a local refinement: the objective is
    one-dimensional and smooth, so this is exact enough and adds no dependency.

    `None` when there is nothing to fit on, or fewer than `MIN_VALIDATION` rows.
    Returning 1.0 instead would report an unfitted model as though it had been
    calibrated, and fitting on four rows would report an artefact as one."""
    if len(rows) < MIN_VALIDATION:
        return None
    best = min(_GRID, key=lambda t: _nll(rows, t))
    fine = [best + 0.002 * i for i in range(-9, 10) if T_MIN <= best + 0.002 * i <= T_MAX]
    return min(fine, key=lambda t: _nll(rows, t))


def apply_temperature(probs, temperature):
    """Softmax of (log p / T). T = 1 is the identity; T > 1 flattens; T < 1
    sharpens. The ranking is untouched at every T, so accuracy cannot move."""
    scaled = {k: _log(v) / temperature for k, v in probs.items()}
    peak = max(scaled.values())
    exponentiated = {k: math.exp(v - peak) for k, v in scaled.items()}
    total = sum(exponentiated.values())
    return {k: v / total for k, v in exponentiated.items()}


def calibrate(validation, test, n_bins=N_BINS):
    """-> {"temperature", "ece_raw", "ece_scaled", "n_validation", "n_test"}.

    `None` when the model returned no probabilities. A model that cannot be
    calibrated is absent from the calibration table rather than scored zero,
    which would read as perfectly calibrated."""
    if not test:
        return None
    temperature = fit_temperature(validation)
    raw_conf = [max(p.values()) for p, _ in test]
    raw_correct = [int(max(p, key=p.get) == gold) for p, gold in test]
    out = {
        "temperature": temperature,
        "ece_raw": ece(raw_conf, raw_correct, n_bins),
        "ece_scaled": None,
        "n_validation": len(validation),
        "n_test": len(test),
        "n_bins": n_bins,
        "min_validation": MIN_VALIDATION,
        "fit_refused": len(validation) < MIN_VALIDATION,
    }
    if temperature is not None:
        scaled = [apply_temperature(p, temperature) for p, _ in test]
        out["ece_scaled"] = ece([max(p.values()) for p in scaled], raw_correct, n_bins)
    return out


def reliability(confidences, correct, n_bins=N_BINS):
    """-> {"bin_centres", "accuracy", "counts"} for the reliability diagram.

    One point per **non-empty** bin: a bin nothing fell into is not a model that
    was wrong there, and plotting it at zero would draw a curve the run never
    measured. The x of each point is the mean confidence actually expressed in
    that bin rather than the nominal centre of the bucket, because the curve is
    read against the diagonal."""
    if len(confidences) != len(correct):
        raise ValueError(f"{len(confidences)} confidences against {len(correct)} outcomes")
    if not confidences:
        return None
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, correct):
        bins[min(n_bins - 1, int(c * n_bins))].append((c, ok))
    filled = [rows for rows in bins if rows]
    return {
        "bin_centres": [sum(c for c, _ in rows) / len(rows) for rows in filled],
        "accuracy": [sum(ok for _, ok in rows) / len(rows) for rows in filled],
        "counts": [len(rows) for rows in filled],
    }

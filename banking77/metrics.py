"""Accuracy, top-5, macro-F1, ECE, a bootstrap CI, one fitted temperature and
McNemar's test. Plain Python and numpy; no scipy, which would be a large
dependency for the forty lines below.

Temperature scaling works on `log(p)` of the recorded distribution. laya has
already divided its logits by its own temperature before the softmax, and the
response never carries the raw logits, but `softmax(log(p) / T)` is exactly the
standard temperature-scaling family applied to those logits: dividing by T
composes with whatever laya already did. The recorded probabilities are rounded
to four decimals by laya, so an exact zero is clipped to 1e-6 before the log.
"""
import math

import numpy as np

PROB_FLOOR = 1e-6


def correctness(pred, gold):
    return [int(p == g) for p, g in zip(pred, gold)]


def accuracy(pred, gold):
    c = correctness(pred, gold)
    return sum(c) / len(c) if c else 0.0


def top_k_accuracy(prob_dicts, gold, k):
    """Ties keep the option order the distribution was recorded in, so a tie never
    flatters the model by accident."""
    hits = 0
    for probs, g in zip(prob_dicts, gold):
        order = sorted(probs, key=lambda label: -probs[label])
        hits += int(g in order[:k])
    return hits / len(gold) if gold else 0.0


def macro_f1(pred, gold, labels):
    total = 0.0
    for label in labels:
        tp = sum(1 for p, g in zip(pred, gold) if p == label and g == label)
        fp = sum(1 for p, g in zip(pred, gold) if p == label and g != label)
        fn = sum(1 for p, g in zip(pred, gold) if p != label and g == label)
        total += 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    return total / len(labels) if labels else 0.0


def ece(confidences, correct, bins=15):
    """Expected calibration error over `bins` equal-width bins on [0, 1]. Empty
    bins contribute nothing; occupied ones are weighted by how many land in them."""
    if not confidences:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    conf = np.asarray(confidences, dtype=float)
    acc = np.asarray(correct, dtype=float)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.sum() * abs(acc[mask].mean() - conf[mask].mean())
    return float(total / len(conf))


def bootstrap_ci(values, n=1000, seed=0, alpha=0.05):
    """Percentile CI of the mean over `n` resamples with replacement."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def to_logits(probs):
    return [math.log(max(p, PROB_FLOOR)) for p in probs]


def apply_temperature(logits, t):
    z = np.asarray(logits, dtype=float) / t
    z = np.exp(z - z.max())
    return (z / z.sum()).tolist()


def fit_temperature(logit_rows, gold_idx, grid=None):
    """One temperature, grid-searched to the lowest negative log likelihood. Fitted
    on the validation split carved out of train, never on test."""
    grid = grid if grid is not None else np.exp(np.linspace(math.log(0.05), math.log(20.0), 241))
    best, best_nll = 1.0, float("inf")
    for t in grid:
        nll = 0.0
        for logits, g in zip(logit_rows, gold_idx):
            p = apply_temperature(logits, t)
            nll -= math.log(max(p[g], PROB_FLOOR))
        if nll < best_nll:
            best, best_nll = float(t), nll
    return best


def mcnemar(correct_a, correct_b):
    """Exact two-sided binomial McNemar on the discordant pairs. `n10` is where A
    was right and B wrong."""
    if len(correct_a) != len(correct_b):
        raise ValueError("McNemar compares two arms over the same examples")
    n10 = sum(1 for a, b in zip(correct_a, correct_b) if a and not b)
    n01 = sum(1 for a, b in zip(correct_a, correct_b) if b and not a)
    n = n10 + n01
    if n == 0:
        return {"n10": 0, "n01": 0, "n_discordant": 0, "p_value": 1.0}
    k = min(n10, n01)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return {"n10": n10, "n01": n01, "n_discordant": n, "p_value": min(1.0, 2 * tail)}

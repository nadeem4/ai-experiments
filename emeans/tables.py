"""The match rule and the speedup at the match.

"Matches" means the mean over seeds of the paired MSE ratio (arm test MSE over
the same seed's Lloyd 25-iteration test MSE) is at most 1.000. An arm that does
not match within the largest epoch count has no speedup, never a number.
"""
import statistics

from .consts import EPOCHS


def _by_seed(records, dataset, arm, setting):
    return {r["seed"]: r for r in records
            if r["dataset"] == dataset and r["arm"] == arm and r["setting"] == setting}


def mse_ratios(records, dataset, arm, setting):
    """-> {seed: arm test MSE / that seed's Lloyd iters=25 test MSE}."""
    mine = _by_seed(records, dataset, arm, setting)
    base = _by_seed(records, dataset, "lloyd", "iters=25")
    return {s: mine[s]["test_mse"] / base[s]["test_mse"] for s in mine if s in base}


def matches(ratios):
    return bool(ratios) and statistics.fmean(ratios.values()) <= 1.000


def epochs_to_match(records, dataset, arm):
    """The smallest of EPOCHS that matches, else None."""
    for e in EPOCHS:
        if matches(mse_ratios(records, dataset, arm, f"epochs={e}")):
            return e
    return None


def speedup(records, dataset, arm):
    """Lloyd's 25-iteration time over the arm's time at the matching epoch
    count, per seed then averaged; None when the arm never matches."""
    e = epochs_to_match(records, dataset, arm)
    if e is None:
        return None
    mine = _by_seed(records, dataset, arm, f"epochs={e}")
    base = _by_seed(records, dataset, "lloyd", "iters=25")
    pairs = [base[s]["train_seconds"] / mine[s]["train_seconds"] for s in mine if s in base]
    return statistics.fmean(pairs)

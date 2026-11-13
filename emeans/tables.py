"""The match rule and the speedup at the match.

"Matches" means the mean over seeds of the paired MSE ratio (arm test MSE over
the same seed's Lloyd 25-iteration test MSE) is at most 1.000. An arm that does
not match within the largest epoch count has no speedup, never a number.
"""
import statistics

import numpy as np

from . import stats
from .consts import EPOCHS


def _by_seed(records, dataset, arm, setting):
    return {r["seed"]: r for r in records
            if r["dataset"] == dataset and r["arm"] == arm and r["setting"] == setting}


def mse_ratios(records, dataset, arm, setting):
    """-> {seed: arm test MSE / that seed's Lloyd iters=25 test MSE}."""
    mine = _by_seed(records, dataset, arm, setting)
    base = _by_seed(records, dataset, "lloyd", "iters=25")
    if not mine or set(mine) != set(base):
        raise ValueError(f"{dataset} {arm} {setting}: seeds {sorted(mine)} do not match "
                         f"Lloyd iters=25 seeds {sorted(base)}; the data is incomplete")
    return {s: mine[s]["test_mse"] / base[s]["test_mse"] for s in mine}


def matches(ratios):
    if not ratios:
        raise ValueError("no paired ratios: missing data is not a mismatch")
    return statistics.fmean(ratios.values()) <= 1.000


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
    pairs = [base[s]["train_seconds"] / mine[s]["train_seconds"] for s in mine]
    return statistics.fmean(pairs)


def ledger(records):
    """-> {arm: [(month, maintenance_s, cumulative_s)]} from part B's per-month
    records. Maintenance is training plus re-assignment, never the add step;
    with several seeds the month's value is their mean. Cumulative is the
    running sum of those monthly values, nothing else."""
    out = {}
    for arm in sorted({r["arm"] for r in records if r["part"] == "b"}):
        by_month = {}
        for r in records:
            if r["part"] == "b" and r["arm"] == arm:
                by_month.setdefault(r["setting"], []).append(r["train_seconds"] + r["assign_seconds"])
        total, rows = 0.0, []
        for month in sorted(by_month):
            monthly = statistics.fmean(by_month[month])
            total += monthly
            rows.append((month, monthly, total))
        out[arm] = rows
    return out


def adding_seconds(records):
    """-> {arm: mean over seeds of the total seconds spent adding docs to the
    index}. Reported apart from the ledger and excluded from every comparison."""
    out = {}
    for arm in sorted({r["arm"] for r in records if r["part"] == "b"}):
        per_seed = {}
        for r in records:
            if r["part"] == "b" and r["arm"] == arm:
                per_seed[r["seed"]] = per_seed.get(r["seed"], 0.0) + r["adding_seconds"]
        out[arm] = statistics.fmean(per_seed.values())
    return out


def pooled_recall(records, arm, nprobe, year="2025"):
    """-> {seed: per-query recall of every query in `year`}, months in order
    (12 months x 500 = 6,000 queries for 2025)."""
    out = {}
    for r in sorted((r for r in records if r["part"] == "b" and r["arm"] == arm
                     and r["setting"].startswith(year)), key=lambda r: r["setting"]):
        out.setdefault(r["seed"], []).extend(r["recall_per_query"][str(nprobe)])
    return {seed: np.array(v) for seed, v in out.items()}


def _paired(records, arm_a, arm_b, nprobe, year):
    a, b = pooled_recall(records, arm_a, nprobe, year), pooled_recall(records, arm_b, nprobe, year)
    if not a or set(a) != set(b):
        raise ValueError(f"{arm_a} seeds {sorted(a)} do not match {arm_b} seeds {sorted(b)}; the data is incomplete")
    return a, b


def end_state(records, arm_a, arm_b, nprobe=16, year="2025"):
    """-> (mean_diff, lo, hi): recall of `arm_a` minus `arm_b`, the per-query
    difference averaged over seeds, then a paired bootstrap over the queries."""
    a, b = _paired(records, arm_a, arm_b, nprobe, year)
    return stats.paired_bootstrap(np.mean(list(a.values()), axis=0), np.mean(list(b.values()), axis=0))


def end_state_per_seed(records, arm_a, arm_b, nprobe=16, year="2025"):
    """-> {seed: mean recall difference}, reported next to the pooled interval."""
    a, b = _paired(records, arm_a, arm_b, nprobe, year)
    return {seed: float((a[seed] - b[seed]).mean()) for seed in a}

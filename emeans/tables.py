"""The match rule and the speedup at the match.

"Matches" means the mean over seeds of the paired MSE ratio (arm test MSE over
the same seed's Lloyd 25-iteration test MSE) is at most 1.000. An arm that does
not match within the largest epoch count has no speedup, never a number.
"""
import csv
import itertools
import json
import statistics
from pathlib import Path

import numpy as np

from . import metrics, stats
from .consts import B_NPROBES, EPOCHS, LLOYD_AT


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
    """-> {arm: [(month, mean_s, min_s, max_s, cumulative_s)]} from part B's
    per-month records. Maintenance is training plus re-assignment, never the add
    step. The month's value is the mean over seeds, with the min and max beside
    it for the month curves; cumulative is the running sum of the means, nothing
    else. A month whose seeds differ from the arm's full seed set raises: a mean
    over fewer seeds is not comparable with its neighbours."""
    out = {}
    for arm in sorted({r["arm"] for r in records if r["part"] == "b"}):
        by_month = {}
        for r in records:
            if r["part"] == "b" and r["arm"] == arm:
                by_month.setdefault(r["setting"], {})[r["seed"]] = r["train_seconds"] + r["assign_seconds"]
        seeds = set().union(*(set(v) for v in by_month.values()))
        total, rows = 0.0, []
        for month in sorted(by_month):
            if set(by_month[month]) != seeds:
                raise ValueError(f"{arm} {month}: seeds {sorted(by_month[month])} differ from the arm's "
                                 f"seeds {sorted(seeds)}; the data is incomplete")
            values = list(by_month[month].values())
            monthly = statistics.fmean(values)
            total += monthly
            rows.append((month, monthly, min(values), max(values), total))
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


QUERIES_PER_MONTH = 500


def pooled_recall(records, arm, nprobe, year="2025"):
    """-> {seed: per-query recall of every query in `year`}, months in order.
    For 2025 each seed must hold all 12 months of exactly 500 queries (6,000),
    or this raises: the bootstrap never runs over fewer queries silently."""
    out, months = {}, {}
    arm_seeds = {r["seed"] for r in records if r["part"] == "b" and r["arm"] == arm}
    for r in sorted((r for r in records if r["part"] == "b" and r["arm"] == arm
                     and r["setting"].startswith(year)), key=lambda r: r["setting"]):
        queries = r["recall_per_query"][str(nprobe)]
        if year == "2025" and len(queries) != QUERIES_PER_MONTH:
            raise ValueError(f"{arm} {r['setting']} seed {r['seed']}: {len(queries)} queries, "
                             f"not {QUERIES_PER_MONTH}")
        out.setdefault(r["seed"], []).extend(queries)
        months.setdefault(r["seed"], []).append(r["setting"])
    expected = [f"2025-{m:02d}" for m in range(1, 13)]
    if year == "2025" and out:
        for seed in sorted(arm_seeds):
            got = months.get(seed, [])
            if got != expected:
                raise ValueError(f"{arm} seed {seed}: 2025 months {got} are not all of {expected[0]}..{expected[-1]}")
    return {seed: np.array(v) for seed, v in out.items()}


def _paired(records, arm_a, arm_b, nprobe, year):
    a, b = pooled_recall(records, arm_a, nprobe, year), pooled_recall(records, arm_b, nprobe, year)
    if not a or set(a) != set(b):
        raise ValueError(f"{arm_a} seeds {sorted(a)} do not match {arm_b} seeds {sorted(b)}; the data is incomplete")
    return a, b


def end_state(records, arm_a, arm_b, nprobe=16, year="2025", n_boot=10_000):
    """-> (mean_diff, lo, hi): recall of `arm_a` minus `arm_b`, the per-query
    difference averaged over seeds, then a paired bootstrap over the queries."""
    a, b = _paired(records, arm_a, arm_b, nprobe, year)
    return stats.paired_bootstrap(np.mean(list(a.values()), axis=0), np.mean(list(b.values()), axis=0),
                                  n_boot=n_boot)


def end_state_per_seed(records, arm_a, arm_b, nprobe=16, year="2025"):
    """-> {seed: mean recall difference}, reported next to the pooled interval."""
    a, b = _paired(records, arm_a, arm_b, nprobe, year)
    return {seed: float((a[seed] - b[seed]).mean()) for seed in a}


# ---- summary.json: every table the report and the site read, from records only ----

def _ci(values):
    mean, lo, hi = metrics.mean_ci(list(values))
    return {"mean": mean, "lo": lo, "hi": hi}


def _match_row(records, dataset, arm):
    """Epochs to match, speedup, and the paired MSE ratio (with its interval) at
    the matching epoch count, or at the largest one when the arm never matches."""
    e = epochs_to_match(records, dataset, arm)
    shown = e if e is not None else max(EPOCHS)
    return {"epochs_to_match": e, "speedup": speedup(records, dataset, arm),
            "ratio": _ci(mse_ratios(records, dataset, arm, f"epochs={shown}").values())}


def _part_a(records, dataset):
    mine = [r for r in records if r["dataset"] == dataset]
    arms = sorted({r["arm"] for r in mine if r["setting"].startswith("epochs=")})
    snaps = _by_seed(mine, dataset, "lloyd", "snapshots")
    base = _by_seed(mine, dataset, "lloyd", "iters=25")
    curve = {}
    if snaps and set(snaps) == set(base):
        curve = {str(i): statistics.fmean(snaps[s]["mse_by_iter"][str(i)] / base[s]["test_mse"] for s in base)
                 for i in LLOYD_AT}
    budget, search = [], []
    cells = sorted({(r["arm"], r["setting"]) for r in mine
                    if r["setting"].startswith("budget=") or ("search" in r and r["setting"].startswith("iters="))})
    for arm, setting in cells:
        rows = list(_by_seed(mine, dataset, arm, setting).values())
        if setting.startswith("budget="):
            budget.append({"arm": arm, "budget_s": rows[0]["budget_s"], "epochs": rows[0]["epochs"],
                           "actual_s": statistics.fmean(r["train_seconds"] for r in rows),
                           "overshoot": any(r["overshoot"] for r in rows),
                           "test_mse": statistics.fmean(r["test_mse"] for r in rows)})
        if "search" in rows[0]:
            search.append({"arm": arm, "setting": setting,
                           "search": {p: {f: statistics.fmean(r["search"][p][f] for r in rows)
                                          for f in rows[0]["search"][p]} for p in rows[0]["search"]}})
    return {"match": {arm: _match_row(mine, dataset, arm) for arm in arms},
            "race": {arm: {str(e): _ci(mse_ratios(mine, dataset, arm, f"epochs={e}").values()) for e in EPOCHS}
                     for arm in arms},
            "lloyd_curve": curve,
            "budget": sorted(budget, key=lambda r: (r["arm"], r["budget_s"])),
            "search": search}


def _part_b(records, n_boot):
    from . import growing

    led = ledger(records)
    adding = adding_seconds(records)
    arms = [a for a in growing.ARMS_B if a in led]
    series = {}
    for arm in arms:
        by_month = {}
        for r in records:
            if r["part"] == "b" and r["arm"] == arm:
                by_month.setdefault(r["setting"], []).append(r)
        months = [row[0] for row in led[arm]]
        per_seed = {}  # seed -> cumulative seconds after each month
        for m in months:
            for r in by_month[m]:
                seq = per_seed.setdefault(r["seed"], [])
                seq.append((seq[-1] if seq else 0.0) + r["train_seconds"] + r["assign_seconds"])
        s = {"months": months, "cumulative_s": [row[4] for row in led[arm]],
             "cumulative_range": {"min": [min(v[i] for v in per_seed.values()) for i in range(len(months))],
                                  "max": [max(v[i] for v in per_seed.values()) for i in range(len(months))]},
             "maintenance_s": {"mean": [row[1] for row in led[arm]], "min": [row[2] for row in led[arm]],
                               "max": [row[3] for row in led[arm]]},
             "imbalance": [statistics.fmean(r["imbalance"] for r in by_month[m]) for m in months],
             "dead": [statistics.fmean(r["dead_clusters"] for r in by_month[m]) for m in months]}
        for p in B_NPROBES:
            v = [[r["search"][str(p)]["recall10"] for r in by_month[m]] for m in months]
            s[f"recall{p}"] = {"mean": [statistics.fmean(x) for x in v], "min": [min(x) for x in v],
                               "max": [max(x) for x in v]}
        series[arm] = s
    end = []
    if any(r["part"] == "b" and r["setting"].startswith("2025") for r in records):
        for a, b in itertools.combinations(arms, 2):
            for p in B_NPROBES:
                mean, lo, hi = end_state(records, a, b, p, n_boot=n_boot)
                end.append({"a": a, "b": b, "nprobe": p, "mean": mean, "lo": lo, "hi": hi,
                            "per_seed": {str(s): v for s, v in end_state_per_seed(records, a, b, p).items()}})
    return {"series": series, "end_state": end,
            "ledger": {arm: {"total_s": led[arm][-1][4], "adding_s": adding[arm]} for arm in arms},
            "facts": {"months": len(led[arms[0]]), "seeds": len({r["seed"] for r in records if r["part"] == "b"})}}


def _part_c(records):
    return {d.rsplit("-d", 1)[1]: _match_row(records, d, "emeans") for d in sorted({r["dataset"] for r in records})}


def summary(records, n_boot=10_000):
    """-> the dict written to results/<tag>/summary.json: per part, only what the
    records hold. Nothing here is typed in; the figures and CSVs read this alone."""
    out = {}
    a = [r for r in records if r["part"] == "a"]
    if a:
        out["a"] = {d: _part_a(a, d) for d in sorted({r["dataset"] for r in a})}
    b = [r for r in records if r["part"] == "b"]
    if b:
        out["b"] = _part_b(b, n_boot)
    c = [r for r in records if r["part"] == "c"]
    if c:
        out["c"] = _part_c(c)
    return out


def write_summary(summary_dict, out_dir):
    """Write summary.json with sorted keys, so a re-run is byte-identical."""
    path = Path(out_dir) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)
    return path


def _blank(v):
    return "" if v is None else v


def write_csv(summary_dict, out_dir):
    """CSVs beside summary.json, written from it and nothing else."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s, paths = summary_dict, []
    if "a" in s:
        paths.append(_csv(out_dir / "a_match.csv", ["dataset", "arm", "epochs_to_match", "speedup", "ratio", "lo", "hi"],
                          [[d, arm, _blank(m["epochs_to_match"]), _blank(m["speedup"]), m["ratio"]["mean"],
                            _blank(m["ratio"]["lo"]), _blank(m["ratio"]["hi"])]
                           for d, part in s["a"].items() for arm, m in part["match"].items()]))
        paths.append(_csv(out_dir / "a_budget.csv",
                          ["dataset", "arm", "budget_s", "epochs", "actual_s", "overshoot", "test_mse"],
                          [[d, r["arm"], r["budget_s"], r["epochs"], r["actual_s"], r["overshoot"], r["test_mse"]]
                           for d, part in s["a"].items() for r in part["budget"]]))
    if "b" in s:
        paths.append(_csv(out_dir / "b_end_state.csv", ["a", "b", "nprobe", "mean", "lo", "hi"],
                          [[e["a"], e["b"], e["nprobe"], e["mean"], e["lo"], e["hi"]] for e in s["b"]["end_state"]]))
        paths.append(_csv(out_dir / "b_ledger.csv", ["arm", "total_s", "adding_s"],
                          [[arm, v["total_s"], v["adding_s"]] for arm, v in s["b"]["ledger"].items()]))
    if "c" in s:
        paths.append(_csv(out_dir / "c_dimension.csv", ["d", "epochs_to_match", "speedup", "ratio"],
                          [[d, _blank(m["epochs_to_match"]), _blank(m["speedup"]), m["ratio"]["mean"]]
                           for d, m in sorted(s["c"].items(), key=lambda kv: int(kv[0]))]))
    return paths

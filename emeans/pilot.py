"""Pilot checks, and the projection of full-run wall clock from the pilot.

The projection assumes training time is linear in n * k * d * epochs, so it is a
planning estimate for the human gate before any full run, not a measurement.
"""
from . import growing, race
from .consts import ARMS, BUDGETS_S, DIMS, EPOCHS, LLOYD_AT


def problems(records, k):
    """Every failed check at once, as readable messages; empty when all is well.
    `k=None` skips the nprobe == k check (full runs do not search at k)."""
    out = []
    with_search = any("search" in r for r in records)
    for r in records:
        unit = f"{r['dataset']} {r['arm']} {r['setting']} seed {r['seed']}"
        missing = [f for f in race.required_fields(r["setting"], with_search) if f not in r]
        if missing:
            out.append(f"{unit}: missing fields {', '.join(missing)}")
        if r["setting"] == "snapshots" and "train_mse_by_iter" in r:
            mses = [r["train_mse_by_iter"][str(i)] for i in LLOYD_AT]
            if any(b > a * (1 + 1e-9) for a, b in zip(mses, mses[1:])):
                out.append(f"{unit}: Lloyd train MSE rises across snapshots")
        search = r.get("search")
        if search is not None and k is not None and str(k) not in search:
            out.append(f"{unit}: search has no nprobe == k ({k}) entry, so exactness is unchecked")
        elif search is not None and k is not None and search[str(k)]["recall10"] < 1.0 - 1e-9:
            out.append(f"{unit}: recall at nprobe == k ({k}) is {search[str(k)]['recall10']}, not 1.0")
        if r["arm"] == "emeans" and r["setting"].startswith("epochs=") and "rows" in r and not r["rows"]:
            out.append(f"{unit}: e-means was not run through begin (no rows recorded)")
    return out


def project(records, target_shape, seeds=5, with_budgets=True, arms=ARMS):
    """-> hours for one dataset at `target_shape = (n, k, d)`: training (from the
    calibration records) and evaluation (re-assignment, MSE, IVF build, search:
    the `eval_seconds` fields) scaled linearly, plus the budget units, whose
    training time is fixed by definition."""
    cal = [r for r in records if r["setting"] == "calibration" and r["arm"] in arms]
    n, k, d = target_shape
    f = lambda r: (n * k * d) / (r["n"] * r["k"] * r["d"])
    spe = {r["arm"]: r["train_seconds"] / r["epochs"] * f(r) for r in cal}
    per_seed = (spe["lloyd"] * (2 * max(LLOYD_AT))  # snapshots chain plus the timed run
                + sum(v for a, v in spe.items() if a != "lloyd") * sum(EPOCHS))
    train_h = (sum(spe.values()) + seeds * per_seed) / 3600

    done = [r for r in records if r["setting"] != "calibration" and "eval_seconds" in r and r["arm"] in arms]
    pilot_seeds = len({r["seed"] for r in done}) or 1
    eval_per_seed = sum(r["eval_seconds"] * f(r) for r in done) / pilot_seeds
    eval_s = seeds * eval_per_seed
    budget = 0.0
    if with_budgets:
        budget = len(arms) * sum(BUDGETS_S) * seeds / 3600
        searched = [r["eval_seconds"] * f(r) for r in done if "search" in r]
        if searched:  # a budget unit is evaluated like an iters=25 unit
            eval_s += len(arms) * len(BUDGETS_S) * seeds * sum(searched) / len(searched)
    eval_h = eval_s / 3600
    return {"shape": {"n": n, "k": k, "d": d}, "seeds": seeds,
            "seconds_per_epoch": spe, "train_hours": train_h, "eval_hours": eval_h,
            "scaled_hours": train_h + eval_h, "budget_hours": budget,
            "hours": train_h + eval_h + budget}


PASSES = {"emeans": 1.5, "lloyd_half": 1.5}  # 0.5 epoch (or half an iteration) of training + 1 re-assignment
YEARLY_PASSES = 26                           # 25 Lloyd iterations + 1 re-assignment, each January


def project_b(records, months, n0, per_month, k, d, queries=500, seeds=3):
    """-> hours for part B at the full timeline, from the pilot's B records.

    Maintenance seconds are turned into a rate per (n * k * d * pass) per arm,
    where a pass is one sweep over the indexed docs, then applied to every
    month's pass count at the index size that month (`n0` docs plus `per_month`
    each month). Evaluation (truth and search) is a rate per
    (n * d * query) and the add step (IVF build, `adding_seconds`) a rate per
    (n * d), both over all pilot records. Truth is cached across arms and
    seeds in the real run, so the evaluation share is an upper bound."""
    b = [r for r in records if r["part"] == "b"]
    rate = {"stale": 0.0}
    for arm, passes in (*PASSES.items(), ("yearly", YEARLY_PASSES)):
        mine = [r for r in b if r["arm"] == arm and (arm != "yearly" or r["setting"].endswith("-01"))]
        work = sum(r["n"] * r["k"] * r["d"] * passes for r in mine)
        rate[arm] = sum(r["train_seconds"] + r["assign_seconds"] for r in mine) / work if work else 0.0
    eval_rate = sum(r["eval_seconds"] for r in b) / sum(r["n"] * r["d"] * r["n_queries"] for r in b)
    sizes = [n0 + per_month * (m + 1) for m in range(months)]
    labels = growing.timeline(months)
    maint = {"stale": 0.0}
    for arm, passes in PASSES.items():
        maint[arm] = seeds * sum(rate[arm] * passes * n * k * d for n in sizes) / 3600
    maint["yearly"] = seeds * sum(rate["yearly"] * YEARLY_PASSES * n * k * d
                                  for n, month in zip(sizes, labels) if month.endswith("-01")) / 3600
    add_rate = sum(r["adding_seconds"] for r in b) / sum(r["n"] * r["d"] for r in b)
    adding_h = seeds * len(growing.ARMS_B) * sum(add_rate * n * d for n in sizes) / 3600
    eval_h = seeds * len(growing.ARMS_B) * sum(eval_rate * n * d * queries for n in sizes) / 3600
    return {"months": months, "seeds": seeds, "rate_per_pass": rate, "maintenance_hours": maint,
            "eval_hours": eval_h, "adding_hours": adding_h,
            "hours": sum(maint.values()) + eval_h + adding_h}


def project_c(records, n, k, dims=DIMS, seeds=3):
    """-> hours for part C: Lloyd and e-means at each dimension, from part A's
    calibration (seconds per epoch scaled by n * k * d). No budgets, no search."""
    per_d = {d: project(records, (n, k, d), seeds, with_budgets=False, arms=("lloyd", "emeans"))["hours"]
             for d in dims}
    return {"per_d": per_d, "hours": sum(per_d.values())}


def problems_b(records, k):
    """Part B's pilot checks: complete records, recall 1.0 at nprobe = k (when
    `k` is given), and e-means and Lloyd-half actually trained (rows touched)."""
    out = []
    for r in records:
        unit = f"{r['dataset']} {r['arm']} {r['setting']} seed {r['seed']}"
        missing = [f for f in growing.REQUIRED if f not in r]
        if missing:
            out.append(f"{unit}: missing fields {', '.join(missing)}")
            continue
        search = r["search"]
        if k is not None and str(k) not in search:
            out.append(f"{unit}: search has no nprobe == k ({k}) entry, so exactness is unchecked")
        elif k is not None and search[str(k)]["recall10"] < 1.0 - 1e-9:
            out.append(f"{unit}: recall at nprobe == k ({k}) is {search[str(k)]['recall10']}, not 1.0")
        if r["arm"] in ("emeans", "lloyd_half") and not r["rows"]:
            out.append(f"{unit}: {r['arm']} touched no rows (e-means was not run through begin)")
    return out

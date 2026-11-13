"""Pilot checks, and the projection of full-run wall clock from the pilot.

The projection assumes training time is linear in n * k * d * epochs, so it is a
planning estimate for the human gate before any full run, not a measurement.
"""
from . import race
from .consts import ARMS, BUDGETS_S, EPOCHS, LLOYD_AT


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


def project(records, target_shape, seeds=5, with_budgets=True):
    """-> hours for one dataset at `target_shape = (n, k, d)`: training (from the
    calibration records) and evaluation (re-assignment, MSE, IVF build, search:
    the `eval_seconds` fields) scaled linearly, plus the budget units, whose
    training time is fixed by definition."""
    cal = [r for r in records if r["setting"] == "calibration"]
    n, k, d = target_shape
    f = lambda r: (n * k * d) / (r["n"] * r["k"] * r["d"])
    spe = {r["arm"]: r["train_seconds"] / r["epochs"] * f(r) for r in cal}
    per_seed = (spe["lloyd"] * (2 * max(LLOYD_AT))  # snapshots chain plus the timed run
                + (spe["emeans"] + spe["minibatch"]) * sum(EPOCHS))
    train_h = (sum(spe.values()) + seeds * per_seed) / 3600

    done = [r for r in records if r["setting"] != "calibration" and "eval_seconds" in r]
    pilot_seeds = len({r["seed"] for r in done}) or 1
    eval_per_seed = sum(r["eval_seconds"] * f(r) for r in done) / pilot_seeds
    eval_s = seeds * eval_per_seed
    budget = 0.0
    if with_budgets:
        budget = len(ARMS) * sum(BUDGETS_S) * seeds / 3600
        searched = [r["eval_seconds"] * f(r) for r in done if "search" in r]
        if searched:  # a budget unit is evaluated like an iters=25 unit
            eval_s += len(ARMS) * len(BUDGETS_S) * seeds * sum(searched) / len(searched)
    eval_h = eval_s / 3600
    return {"shape": {"n": n, "k": k, "d": d}, "seeds": seeds,
            "seconds_per_epoch": spe, "train_hours": train_h, "eval_hours": eval_h,
            "scaled_hours": train_h + eval_h, "budget_hours": budget,
            "hours": train_h + eval_h + budget}

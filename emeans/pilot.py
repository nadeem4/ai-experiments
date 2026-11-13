"""Pilot checks, and the projection of full-run wall clock from the pilot.

The projection assumes training time is linear in n * k * d * epochs, so it is a
planning estimate for the human gate before any full run, not a measurement.
"""
from . import race
from .consts import ARMS, BUDGETS_S, EPOCHS, LLOYD_AT


def problems(records, k):
    """Every failed check at once, as readable messages; empty when all is well."""
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
        search = r.get("search") or {}
        if str(k) in search and search[str(k)]["recall10"] < 1.0 - 1e-9:
            out.append(f"{unit}: recall at nprobe == k ({k}) is {search[str(k)]['recall10']}, not 1.0")
        if r["arm"] == "emeans" and r["setting"].startswith("epochs=") and "rows" in r and not r["rows"]:
            out.append(f"{unit}: e-means was not run through begin (no rows recorded)")
    return out


def project(records, target_shape, seeds=5, with_budgets=True):
    """-> hours for one dataset at `target_shape = (n, k, d)`: the epoch-based
    units scaled linearly from the calibration records, plus the budget units,
    whose time is fixed by definition."""
    cal = [r for r in records if r["setting"] == "calibration"]
    n, k, d = target_shape
    spe = {r["arm"]: r["train_seconds"] / r["epochs"] * (n * k * d) / (r["n"] * r["k"] * r["d"])
           for r in cal}
    epochs_all = sum(EPOCHS)
    per_seed = (spe["lloyd"] * (2 * max(LLOYD_AT))  # snapshots chain plus the timed run
                + (spe["emeans"] + spe["minibatch"]) * epochs_all)
    scaled = (sum(spe.values()) + seeds * per_seed) / 3600
    budget = len(ARMS) * sum(BUDGETS_S) * seeds / 3600 if with_budgets else 0.0
    return {"shape": {"n": n, "k": k, "d": d}, "seeds": seeds,
            "seconds_per_epoch": spe, "scaled_hours": scaled, "budget_hours": budget,
            "hours": scaled + budget}

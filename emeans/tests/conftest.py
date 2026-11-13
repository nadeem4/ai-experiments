"""A results-shaped record set for the summary, table and figure tests: part A
(a match for e-means at 3 epochs, none for MiniBatch, one overshoot), part C at
two dimensions, and part B over two 2020 months and all of 2025, two seeds."""
import numpy as np
import pytest

SEARCH = {"16": {"recall10": 0.8, "scanned_mean": 100.0, "scanned_p95": 150.0, "qps": 1000.0},
          "64": {"recall10": 0.95, "scanned_mean": 400.0, "scanned_p95": 500.0, "qps": 300.0}}
LLOYD_MSE = (10.0, 10.0)
LLOYD_S = (100.0, 200.0)
EMEANS_MSE = {1: 12.0, 2: 10.5, 3: 9.9, 5: 9.5}


def _a(dataset, arm, setting, seed, mse, seconds, part="a", **extra):
    return {"part": part, "dataset": dataset, "arm": arm, "setting": setting, "seed": seed,
            "test_mse": mse, "train_seconds": seconds, "epochs": 1, **extra}


def part_a(dataset="sift", part="a", with_budgets=True):
    out = []
    for s in (0, 1):
        out.append({"part": part, "dataset": dataset, "arm": "lloyd", "setting": "snapshots", "seed": s,
                    "mse_by_iter": {str(i): 10.0 + 25 / i for i in (1, 2, 3, 5, 10, 25)}})
        out.append(_a(dataset, "lloyd", "iters=25", s, LLOYD_MSE[s], LLOYD_S[s], part, search=SEARCH))
        for e, mse in EMEANS_MSE.items():
            out.append(_a(dataset, "emeans", f"epochs={e}", s, mse, 10.0 * e + 10 * s, part))
            out.append(_a(dataset, "minibatch", f"epochs={e}", s, 11.0, 5.0 * e, part))
        if with_budgets:
            for arm in ("lloyd", "emeans", "minibatch"):
                for b in (10, 30):
                    over = arm == "emeans" and b == 10
                    out.append(_a(dataset, arm, f"budget={b}s", s, 10.0, b * (1.2 if over else 0.9), part,
                                  budget_s=b, overshoot=over, epochs=b // 5, search=SEARCH))
    return out


def part_b(arms=("stale", "yearly", "emeans", "lloyd_half"), seeds=(0, 1)):
    out = []
    months = ["2020-01", "2020-02"] + [f"2025-{m:02d}" for m in range(1, 13)]
    for arm_i, arm in enumerate(arms):
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for month in months:
                q = rng.random(500)
                maint = 0.0 if arm == "stale" else 1.0 + seed
                out.append({
                    "part": "b", "dataset": "arxiv", "arm": arm, "setting": month, "seed": seed,
                    "train_seconds": maint, "assign_seconds": 0.5 if maint else 0.0,
                    "adding_seconds": 0.1, "maintenance_seconds": maint, "imbalance": 2.0 + arm_i,
                    "dead_clusters": arm_i,
                    "search": {"16": {"recall10": float((q + 0.02 * arm_i).mean())},
                               "64": {"recall10": float((q + 0.02 * arm_i).mean())}},
                    "recall_per_query": {"16": (q + 0.02 * arm_i).tolist(), "64": (q + 0.02 * arm_i).tolist()}})
    return out


@pytest.fixture
def fake_records():
    return (part_a() + part_b()
            + [r for d in ("arxiv-d64", "arxiv-d128") for r in part_a(d, "c", False) if r["arm"] != "minibatch"])

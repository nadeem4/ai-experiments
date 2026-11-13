"""The paired bootstrap, the end-state comparison and the B/C projections."""
import numpy as np
import pytest

from emeans import growing, pilot, stats, tables
from emeans.consts import DIMS


def test_identical_arrays_give_zero_and_a_zero_interval():
    a = np.random.default_rng(0).random(200)
    assert stats.paired_bootstrap(a, a.copy()) == (0.0, 0.0, 0.0)


def test_a_constant_offset_is_recovered_and_excludes_zero():
    a = np.random.default_rng(0).random(200)
    mean, lo, hi = stats.paired_bootstrap(a + 0.1, a)
    assert mean == pytest.approx(0.1) and lo > 0


def test_pairing_matters():
    """A small steady gain buried in large per-query variation: unpaired spans 0, paired does not."""
    rng = np.random.default_rng(1)
    b = rng.normal(0, 1, 400)
    a = b + 0.1 + rng.normal(0, 0.01, 400)
    mean, lo, hi = stats.paired_bootstrap(a, b)
    assert lo > 0
    rng2 = np.random.default_rng(0)
    unpaired = [rng2.choice(a, 400).mean() - rng2.choice(b, 400).mean() for _ in range(2000)]
    assert np.percentile(unpaired, 2.5) < 0 < np.percentile(unpaired, 97.5)


def test_the_bootstrap_is_seeded():
    rng = np.random.default_rng(2)
    a, b = rng.random(100), rng.random(100)
    assert stats.paired_bootstrap(a, b, seed=3) == stats.paired_bootstrap(a, b, seed=3)


def b_records(seeds=(0, 1, 2), gain=0.05):
    """12 months of 2025 x 500 queries per arm and seed; 'emeans' beats 'stale' by `gain` per query."""
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for m in range(1, 13):
            base = rng.random(500)
            for arm, offset in (("stale", 0.0), ("emeans", gain)):
                out.append({"part": "b", "dataset": "arxiv", "arm": arm, "setting": f"2025-{m:02d}",
                            "seed": seed, "recall_per_query": {"16": (base + offset).tolist()}})
    # an earlier year that must be ignored
    out += [{"part": "b", "dataset": "arxiv", "arm": a, "setting": "2024-12", "seed": 0,
             "recall_per_query": {"16": [0.0] * 500}} for a in ("stale", "emeans")]
    return out


def test_end_state_pools_exactly_6000_queries_of_2025():
    pooled = tables.pooled_recall(b_records(), "emeans", 16)
    assert set(pooled) == {0, 1, 2} and all(len(v) == 6000 for v in pooled.values())


def test_end_state_is_a_paired_bootstrap_over_seed_averaged_queries():
    mean, lo, hi = tables.end_state(b_records(), "emeans", "stale")
    assert mean == pytest.approx(0.05) and lo > 0
    assert tables.end_state_per_seed(b_records(), "emeans", "stale") == pytest.approx({0: 0.05, 1: 0.05, 2: 0.05})


def test_end_state_refuses_unequal_seeds():
    recs = [r for r in b_records() if not (r["arm"] == "stale" and r["seed"] == 2)]
    with pytest.raises(ValueError, match="seeds"):
        tables.end_state(recs, "emeans", "stale")


def pilot_b_records():
    out = []
    for arm, maint in (("stale", 0.0), ("yearly", 26.0), ("emeans", 1.5), ("lloyd_half", 1.5)):
        for m, month in enumerate(growing.timeline(3)):
            seconds = maint if (arm != "yearly" or month.endswith("-01")) else 0.0
            out.append({"part": "b", "dataset": "arxiv", "arm": arm, "setting": month, "seed": 0,
                        "n": 1000 + 10 * m, "k": 8, "d": 4, "n_queries": 10,
                        "train_seconds": seconds * 0.8, "assign_seconds": seconds * 0.2, "eval_seconds": 0.5})
    return out


def test_project_b_scales_linearly_with_the_number_of_months():
    one = pilot.project_b(pilot_b_records(), months=12, n0=1000, per_month=0, k=8, d=4, queries=10, seeds=1)
    two = pilot.project_b(pilot_b_records(), months=24, n0=1000, per_month=0, k=8, d=4, queries=10, seeds=1)
    assert two["hours"] == pytest.approx(2 * one["hours"]) and one["hours"] > 0


def test_project_b_uses_the_policys_pass_counts():
    p = pilot.project_b(pilot_b_records(), months=12, n0=1000, per_month=0, k=8, d=4, queries=10, seeds=1)
    assert p["maintenance_hours"]["stale"] == 0
    # emeans: 12 months at 1.5 passes; yearly: one January at 26 passes (rates are per pass)
    assert p["maintenance_hours"]["emeans"] == pytest.approx(12 * 1.5 * p["rate_per_pass"]["emeans"] / 3600
                                                              * 1000 * 8 * 4)
    assert p["maintenance_hours"]["yearly"] == pytest.approx(26 * p["rate_per_pass"]["yearly"] / 3600
                                                             * 1000 * 8 * 4)


def a_calibration_records():
    return [{"part": "a", "dataset": "sift", "arm": arm, "setting": "calibration", "seed": 0, "n": 1000,
             "k": 8, "d": 4, "train_seconds": 1.0 + i, "epochs": 1}
            for i, arm in enumerate(("lloyd", "emeans", "minibatch"))]


def test_project_c_covers_lloyd_and_emeans_only_and_grows_with_d():
    recs = a_calibration_records()
    wide = pilot.project_c(recs, n=1000, k=8, dims=DIMS, seeds=3)
    narrow = pilot.project_c(recs, n=1000, k=8, dims=DIMS[:2], seeds=3)
    assert wide["hours"] > narrow["hours"] > 0 and set(wide["per_d"]) == set(DIMS)
    no_mb = [r for r in recs if r["arm"] != "minibatch"]
    assert pilot.project_c(no_mb, n=1000, k=8, dims=DIMS, seeds=3)["hours"] == pytest.approx(wide["hours"])

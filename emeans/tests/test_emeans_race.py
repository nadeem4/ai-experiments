"""The race runner on synthetic 3,000 x 8 data, k = 16, one thread."""
import json

import numpy as np
import pytest

from emeans import data, env, race, store
from emeans.consts import ARMS, EPOCHS, LLOYD_AT
from emeans.experiment import Scale

SCALE = Scale(n=None, n_test=500, n_queries=50, k=16, seeds_a=2, seeds_bc=1, b_months=3,
              nprobes=(1, 4, 16))


@pytest.fixture(autouse=True)
def one_thread():
    before = env.faiss.omp_get_max_threads()
    env.pin_threads(1)
    yield
    env.pin_threads(before)


@pytest.fixture(scope="module")
def split():
    rng = np.random.default_rng(0)
    centres = rng.random((16, 8), dtype=np.float32) * 10
    x = (centres[rng.integers(0, 16, 3000)] + rng.normal(0, 0.5, (3000, 8))).astype(np.float32)
    return data.make_split(x, n_test=500, n_queries=50)


def small_units(with_budgets=False):
    return race.units("syn", ARMS, 2, with_budgets=with_budgets)


def test_units_lists_calibration_first_then_every_arm_setting_per_seed():
    us = race.units("syn", ARMS, 2, with_budgets=True)
    assert [u[3] for u in us[:3]] == ["calibration"] * 3 and {u[2] for u in us[:3]} == set(ARMS)
    assert all(len(u) == 5 and u[0] == "a" and u[1] == "syn" for u in us)
    assert len(set(us)) == len(us)
    for seed in (0, 1):
        mine = {(u[2], u[3]) for u in us if u[4] == seed and u[3] != "calibration"}
        assert ("lloyd", "snapshots") in mine and ("lloyd", "iters=25") in mine
        for arm in ("emeans", "minibatch"):
            assert {(arm, f"epochs={e}") for e in EPOCHS} <= mine
        assert {(a, "budget=10s") for a in ARMS} <= mine
    assert not [u for u in race.units("syn", ARMS, 2, with_budgets=False) if "budget" in u[3]]


def test_run_writes_one_complete_record_per_unit(split, tmp_path):
    path = tmp_path / "a.jsonl"
    us = small_units()
    race.run(us, split, SCALE, path, threads=1)
    records = store.load(path)
    assert [store.key(r) for r in records] == us
    for r in records:
        missing = [f for f in race.required_fields(r["setting"]) if f not in r]
        assert not missing, (r["setting"], missing)
        assert r["threads"] == 1 and r["observed_threads"] <= 1
    json.dumps(records)  # all of it is JSON


def test_rerunning_appends_nothing(split, tmp_path):
    path = tmp_path / "a.jsonl"
    race.run(small_units(), split, SCALE, path, threads=1)
    before = path.read_bytes()
    race.run(small_units(), split, SCALE, path, threads=1)
    assert path.read_bytes() == before


def test_a_killed_run_resumes_with_exactly_the_missing_units(split, tmp_path):
    path = tmp_path / "a.jsonl"
    us = small_units()
    race.run(us, split, SCALE, path, threads=1)
    lines = path.read_bytes().splitlines(keepends=True)
    path.write_bytes(b"".join(lines[:-2]) + lines[-2][:25])  # cut mid-line
    race.run(us, split, SCALE, path, threads=1)
    assert [store.key(r) for r in store.load(path)] == us[:-2] + us[-2:]


def test_search_only_for_iters_25_and_budget_units(split, tmp_path):
    path = tmp_path / "a.jsonl"
    race.run(small_units(), split, SCALE, path, threads=1)
    for r in store.load(path):
        has = "search" in r
        assert has == (r["setting"].startswith("iters=") or r["setting"].startswith("budget="))
    lloyd = next(r for r in store.load(path) if r["setting"] == "iters=25" and r["seed"] == 0)
    assert set(lloyd["search"]) == {"1", "4", "16"}
    assert lloyd["search"]["16"]["recall10"] == 1.0  # nprobe == k scans everything


def test_search_can_be_switched_off(split, tmp_path):
    path = tmp_path / "a.jsonl"
    race.run(small_units(), split, SCALE, path, threads=1, with_search=False)
    assert all("search" not in r for r in store.load(path))


def test_snapshots_record_mse_by_iteration_and_no_timing(split, tmp_path):
    path = tmp_path / "a.jsonl"
    race.run(small_units(), split, SCALE, path, threads=1)
    snap = next(r for r in store.load(path) if r["setting"] == "snapshots")
    assert sorted(map(int, snap["mse_by_iter"])) == sorted(LLOYD_AT)
    assert sorted(map(int, snap["train_mse_by_iter"])) == sorted(LLOYD_AT)
    assert "train_seconds" not in snap


def test_a_tiny_budget_gets_one_epoch_and_is_flagged(split, tmp_path):
    path = tmp_path / "a.jsonl"
    cal = [u for u in small_units() if u[3] == "calibration"]
    race.run(cal + [("a", "syn", "emeans", "budget=0s", 0)], split, SCALE, path, threads=1)
    r = store.load(path)[-1]
    assert r["epochs"] == 1 and r["overshoot"] is True and r["budget_s"] == 0


def test_a_budget_unit_without_calibration_names_the_arm(split, tmp_path):
    with pytest.raises(SystemExit, match="minibatch"):
        race.run([("a", "syn", "minibatch", "budget=10s", 0)], split, SCALE,
                 tmp_path / "a.jsonl", threads=1)


def test_test_mse_is_evaluated_on_the_test_set_only(split, tmp_path):
    held_out = tmp_path / "a.jsonl"
    leaky = tmp_path / "b.jsonl"
    us = [u for u in small_units() if u[3] == "calibration" or u[3] == "epochs=1"][:4]
    leaky_split = split._replace(test=split.train[:500].copy())
    race.run(us, split, SCALE, held_out, threads=1)
    race.run(us, leaky_split, SCALE, leaky, threads=1)
    a = [r["test_mse"] for r in store.load(held_out) if "test_mse" in r]
    b = [r["test_mse"] for r in store.load(leaky) if "test_mse" in r]
    assert a and a != b


def test_a_pool_wider_than_the_pinned_count_is_refused(split):
    env.pin_threads(2)
    with pytest.raises(SystemExit, match="threads"):
        race.run_unit(small_units()[0], split, SCALE, {}, threads=1)

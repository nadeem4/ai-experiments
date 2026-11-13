"""Pilot checks and the wall-clock projection."""
import json

import numpy as np
import pytest

from emeans import data, env, experiment, pilot, race, store
from emeans.consts import ARMS
from emeans.experiment import Scale

SCALE = Scale(n=None, n_test=500, n_queries=50, k=16, seeds_a=1, seeds_bc=1, b_months=3,
              nprobes=(1, 4, 16))


@pytest.fixture(autouse=True)
def one_thread():
    before = env.faiss.omp_get_max_threads()
    env.pin_threads(1)
    yield
    env.pin_threads(before)


@pytest.fixture(scope="module")
def good_records(tmp_path_factory):
    rng = np.random.default_rng(0)
    centres = rng.random((16, 8), dtype=np.float32) * 10
    x = (centres[rng.integers(0, 16, 3000)] + rng.normal(0, 0.5, (3000, 8))).astype(np.float32)
    path = tmp_path_factory.mktemp("pilot") / "a.jsonl"
    race.run(race.units("syn", ARMS, 1), data.make_split(x, n_test=500, n_queries=50), SCALE, path, threads=1)
    return store.load(path)


def test_good_records_have_no_problems(good_records):
    assert pilot.problems(good_records, 16) == []


def test_lloyd_train_mse_that_rises_is_reported(good_records):
    recs = [dict(r) for r in good_records]
    snap = next(r for r in recs if r["setting"] == "snapshots")
    snap["train_mse_by_iter"] = {**snap["train_mse_by_iter"], "25": 1e9}
    out = pilot.problems(recs, 16)
    assert len(out) == 1 and "rise" in out[0]


def test_recall_below_one_at_nprobe_k_is_reported(good_records):
    recs = [dict(r) for r in good_records]
    r = next(r for r in recs if r["setting"] == "iters=25")
    r["search"] = {**r["search"], "16": {**r["search"]["16"], "recall10": 0.9}}
    out = pilot.problems(recs, 16)
    assert len(out) == 1 and "nprobe" in out[0]


def test_a_unit_missing_a_field_is_reported(good_records):
    recs = [dict(r) for r in good_records]
    next(r for r in recs if r["setting"] == "epochs=2" and r["arm"] == "minibatch").pop("imbalance")
    out = pilot.problems(recs, 16)
    assert len(out) == 1 and "imbalance" in out[0]


def test_emeans_not_run_through_begin_is_reported(good_records):
    recs = [dict(r) for r in good_records]
    next(r for r in recs if r["arm"] == "emeans" and r["setting"] == "epochs=3")["rows"] = 0
    out = pilot.problems(recs, 16)
    assert len(out) == 1 and "begin" in out[0]


def test_project_scales_seconds_per_epoch_by_n_k_d(good_records):
    one = pilot.project(good_records, (3000, 16, 8))
    big = pilot.project(good_records, (6000, 16, 8))
    assert big["scaled_hours"] == pytest.approx(2 * one["scaled_hours"])
    assert big["budget_hours"] == one["budget_hours"] and one["hours"] > 0


def test_project_without_budgets_adds_none(good_records):
    assert pilot.project(good_records, (3000, 16, 8), with_budgets=False)["budget_hours"] == 0


def test_a_pilot_run_writes_the_projection_with_all_four_keys(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    x = rng.random((3000, 8), dtype=np.float32)
    monkeypatch.setattr(experiment, "PILOT", SCALE)
    monkeypatch.setattr(experiment, "load_split", lambda args, scale: data.make_split(x, n_test=500, n_queries=50))
    args = experiment_args(tmp_path, ["--tag", "pilot"])
    experiment.run(args)
    proj = json.loads((tmp_path / "results" / "pilot" / "projection.json").read_text())
    assert set(proj) == {"a_sift", "a_arxiv", "b", "c"} and proj["b"] is None and proj["c"] is None
    assert proj["a_sift"]["hours"] > 0 and proj["a_arxiv"]["extrapolated"] is True


def experiment_args(out, argv):
    import argparse

    import cli

    p = argparse.ArgumentParser()
    cli.add_standard_arguments(p)
    experiment.add_run_arguments(p)
    return p.parse_args(argv + ["--out", str(out)])


def test_report_exits_non_zero_on_problems(tmp_path):
    args = experiment_args(tmp_path, ["--tag", "pilot"])
    bad = {"part": "a", "dataset": "sift", "arm": "lloyd", "setting": "iters=25", "seed": 0}
    store.append(tmp_path / "runs" / "pilot" / "a.jsonl", bad)
    with pytest.raises(SystemExit) as e:
        experiment.report(args)
    assert e.value.code != 0

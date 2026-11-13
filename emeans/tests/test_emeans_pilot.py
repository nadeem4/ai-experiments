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


def test_project_grows_with_evaluation_time(good_records):
    assert all(r["eval_seconds"] > 0 for r in good_records if r["setting"] != "calibration")
    base = pilot.project(good_records, (3000, 16, 8))
    slower = [dict(r, eval_seconds=r["eval_seconds"] + 100) if "eval_seconds" in r else r
              for r in good_records]
    more = pilot.project(slower, (3000, 16, 8))
    assert more["scaled_hours"] > base["scaled_hours"] and more["eval_hours"] > base["eval_hours"]


def test_a_search_without_nprobe_k_is_reported(good_records):
    recs = [dict(r) for r in good_records]
    r = next(r for r in recs if r["setting"] == "iters=25")
    r["search"] = {p: v for p, v in r["search"].items() if p != "16"}
    out = pilot.problems(recs, 16)
    assert len(out) == 1 and "nprobe == k" in out[0]


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
    assert set(proj) == {"a_sift", "a_arxiv", "b", "c"} and proj["b"] is None and proj["c"]["hours"] > 0
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


# ---- parts B and C through the cli ----

from emeans import growing  # noqa: E402

B_SCALE = Scale(n=300, n_test=200, n_queries=10, k=8, seeds_a=1, seeds_bc=1, b_months=3, nprobes=(1, 8))


def synthetic_vectors(path):
    rng = np.random.default_rng(0)
    months = np.array(["2019-12"] * 400 + [m for m in growing.timeline(3) for _ in range(600)])
    x = rng.normal(size=(len(months), 8)).astype(np.float32)
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", x)
    np.save(path / "months.npy", months)


def test_a_part_b_pilot_runs_checks_and_projects(tmp_path, monkeypatch):
    synthetic_vectors(tmp_path / "vec")
    monkeypatch.setattr(experiment, "PILOT", B_SCALE)
    args = experiment_args(tmp_path, ["--tag", "pilot", "--part", "b", "--data-dir", str(tmp_path / "vec")])
    experiment.run(args)
    records = store.load(tmp_path / "runs" / "pilot" / "b.jsonl")
    assert len(records) == 12 and {len(r["search"]) for r in records} == {3}  # 16, 64 and nprobe = k
    experiment.report(args)  # no problems: does not exit
    results = tmp_path / "results" / "pilot"
    assert json.loads((results / "summary.json").read_text())["b"]["facts"]["months"] == 3
    assert (results / "figures" / "b-recall-over-time.png").exists() and (results / "b_ledger.csv").exists()
    proj = json.loads((tmp_path / "results" / "pilot" / "projection.json").read_text())
    assert proj["b"]["hours"] > 0 and proj["b"]["months"] == 72


def test_a_part_c_pilot_runs_every_dimension_and_reports_the_match(tmp_path, monkeypatch, capsys):
    synthetic_vectors(tmp_path / "vec")
    monkeypatch.setattr(experiment, "PILOT", B_SCALE)
    monkeypatch.setattr(experiment, "DIMS", (4, 8))
    args = experiment_args(tmp_path, ["--tag", "pilot", "--part", "c", "--data-dir", str(tmp_path / "vec")])
    experiment.run(args)
    records = store.load(tmp_path / "runs" / "pilot" / "c.jsonl")
    assert {r["dataset"] for r in records} == {"arxiv-d4", "arxiv-d8"}
    experiment.report(args)
    out = capsys.readouterr().out
    assert "arxiv-d4" in out and "epochs to match" in out
    assert set(json.loads((tmp_path / "results" / "pilot" / "summary.json").read_text())["c"]) == {"4", "8"}


def test_problems_b_flags_missing_fields_recall_below_one_and_unrun_emeans():
    good = {f: 1 for f in growing.REQUIRED}
    good.update(arm="emeans", setting="2020-01", seed=0, search={"8": {"recall10": 1.0}}, rows=100,
                part="b", dataset="arxiv")
    assert pilot.problems_b([good], 8) == []
    assert "missing" in pilot.problems_b([{k: v for k, v in good.items() if k != "imbalance"}], 8)[0]
    assert "nprobe" in pilot.problems_b([dict(good, search={"8": {"recall10": 0.5}})], 8)[0]
    assert "begin" in pilot.problems_b([dict(good, rows=0)], 8)[0]


def test_problems_b_flags_a_stale_arm_that_spent_time_and_an_emeans_warm_start_that_did_not_move():
    base = {f: 1 for f in growing.REQUIRED}
    base.update(part="b", dataset="arxiv", seed=0, search={"8": {"recall10": 1.0}}, rows=100)
    stale = dict(base, arm="stale", setting="2020-01", maintenance_seconds=0.5)
    assert "maintenance" in pilot.problems_b([stale], 8)[0]
    ok = dict(stale, maintenance_seconds=0)
    assert pilot.problems_b([ok], 8) == []
    e1 = dict(base, arm="emeans", setting="2020-01", centroids_hash="a")
    e2 = dict(base, arm="emeans", setting="2020-02", centroids_hash="a")
    assert "did not move" in pilot.problems_b([e1, e2], 8)[0]
    assert pilot.problems_b([e1, dict(e2, centroids_hash="b")], 8) == []
    other_seed = dict(e2, seed=1)  # a different chain is not the previous month
    assert pilot.problems_b([e1, other_seed], 8) == []

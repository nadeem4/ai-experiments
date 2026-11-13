"""The emeans experiment's `cli` contract and its guard rails."""
import argparse
import json

import pytest

import cli
from emeans import experiment, store


def parser():
    p = argparse.ArgumentParser()
    cli.add_standard_arguments(p)
    experiment.add_run_arguments(p)
    return p


def test_tags_is_empty_with_no_results(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, "EXPERIMENT_DIR", tmp_path)
    assert experiment.tags() == []


def test_tags_lists_result_folders(tmp_path, monkeypatch):
    (tmp_path / "results" / "pilot").mkdir(parents=True)
    monkeypatch.setattr(experiment, "EXPERIMENT_DIR", tmp_path)
    assert experiment.tags() == ["pilot"]


def test_part_b_is_accepted_and_d_is_rejected():
    assert parser().parse_args(["--part", "b"]).part == "b"
    with pytest.raises(SystemExit):
        parser().parse_args(["--part", "d"])


def test_part_defaults_to_a_and_paths_have_defaults():
    args = parser().parse_args([])
    assert args.part == "a"
    assert args.cache_dir is None and args.data_dir is None


def test_it_is_registered_and_costs_nothing():
    assert "emeans" in cli.EXPERIMENTS
    assert experiment.NAME == "emeans" and experiment.COSTS_MONEY is False


def test_scale_follows_the_tag():
    assert experiment.scale_for("pilot") is experiment.PILOT
    assert experiment.scale_for("full") is experiment.FULL
    assert experiment.PILOT.k == 256 and experiment.FULL.k == 4096


def test_any_other_tag_is_refused():
    with pytest.raises(SystemExit):
        experiment.scale_for("fulll")


def test_dry_run_part_b_lists_the_pilot_chains_and_reads_no_data(tmp_path, capsys):
    experiment.run(parser().parse_args(["--tag", "pilot", "--dry-run", "--part", "b", "--out", str(tmp_path)]))
    out = capsys.readouterr().out
    assert "part b" in out and "12 units" in out
    assert "stale 2020-01" in out and "lloyd_half 2020-03" in out and "2020-04" not in out
    assert not (tmp_path / "runs").exists()


def test_dry_run_part_c_lists_every_dimension(tmp_path, capsys):
    experiment.run(parser().parse_args(["--tag", "pilot", "--dry-run", "--part", "c", "--out", str(tmp_path)]))
    out = capsys.readouterr().out
    assert "arxiv-d64 lloyd calibration" in out and "arxiv-d768 emeans epochs=3" in out
    assert "minibatch" not in out and "budget=" not in out


def test_dry_run_lists_part_a_units_and_runs_nothing(tmp_path, capsys):
    args = parser().parse_args(["--tag", "pilot", "--dry-run", "--out", str(tmp_path)])
    experiment.run(args)
    experiment.report(args)  # a dry run reports nothing
    out = capsys.readouterr().out
    assert "lloyd calibration" in out and "emeans epochs=3" in out and "lloyd iters=25" in out
    assert "budget=" not in out  # the pilot leaves budgets out
    assert not (tmp_path / "runs").exists()


def test_full_dry_run_lists_budget_units(tmp_path, capsys):
    experiment.run(parser().parse_args(["--tag", "full", "--dry-run", "--out", str(tmp_path)]))
    assert "minibatch budget=120s" in capsys.readouterr().out


@pytest.mark.parametrize("part", ["b", "c"])
def test_parts_b_and_c_without_vectors_say_how_to_fetch_them(part, tmp_path):
    args = parser().parse_args(["--tag", "pilot", "--part", part, "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="vectors.npy"):
        experiment.run(args)


def test_the_pilot_searches_at_nprobe_k():
    assert experiment.PILOT.k in experiment.PILOT.nprobes


def test_a_missing_sift_file_says_where_to_get_it(tmp_path):
    args = parser().parse_args(["--cache-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="ann-benchmarks"):
        experiment.load_split(args, experiment.PILOT)


def test_dataset_stub_does_not_overwrite_the_card():
    with pytest.raises(SystemExit):
        experiment.dataset()


def test_write_results_builds_summary_csvs_and_figures_from_the_stores(tmp_path, fake_records):
    args = parser().parse_args(["--tag", "full", "--out", str(tmp_path)])
    for part in "abc":
        for r in (r for r in fake_records if r["part"] == part):
            store.append(tmp_path / "runs" / "full" / f"{part}.jsonl", r)
    experiment.write_results(args)
    results = tmp_path / "results" / "full"
    summary = json.loads((results / "summary.json").read_text())
    assert set(summary) == {"a", "b", "c"} and summary["a"]["sift"]["match"]["emeans"]["epochs_to_match"] == 3
    assert (results / "a_match.csv").exists()
    assert {p.name for p in (results / "figures").glob("*.png")} >= {"a-race.png", "b-end-state-gap.png", "c-dimension.png"}


def test_write_results_names_incomplete_data_instead_of_a_traceback(tmp_path, fake_records):
    args = parser().parse_args(["--tag", "full", "--out", str(tmp_path)])
    for r in (r for r in fake_records if r["part"] == "a" and not (r["arm"] == "emeans" and r["seed"] == 1)):
        store.append(tmp_path / "runs" / "full" / "a.jsonl", r)
    with pytest.raises(SystemExit, match="incomplete"):
        experiment.write_results(args)

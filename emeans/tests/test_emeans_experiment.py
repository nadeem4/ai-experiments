"""The emeans experiment's `cli` contract and its guard rails."""
import argparse

import pytest

import cli
from emeans import experiment


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


def test_dry_run_lists_units_and_runs_nothing(capsys):
    args = parser().parse_args(["--tag", "pilot", "--dry-run", "--part", "b"])
    experiment.run(args)
    assert "part b" in capsys.readouterr().out


def test_a_real_run_is_not_built_yet():
    args = parser().parse_args(["--tag", "pilot"])
    with pytest.raises(SystemExit):
        experiment.run(args)


def test_dataset_stub_does_not_overwrite_the_card():
    with pytest.raises(SystemExit):
        experiment.dataset()

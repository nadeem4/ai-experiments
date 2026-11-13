"""The CLI: what `cli` promises every experiment, and what it asks of them.

Two verbs and one contract. The tests that matter are about the contract holding
for all three experiments at once, because the point of the CLI is that a fourth
experiment costs no code here.
"""
import pytest

import cli


class TestTheRegistry:
    def test_it_names_every_experiment_in_the_repository(self):
        assert set(cli.EXPERIMENTS) == {"rerank", "emeans"}

    def test_an_unknown_name_says_what_does_exist(self):
        """Better than a traceback: the answer to 'what can I run' is the error."""
        with pytest.raises(SystemExit) as e:
            cli.load("rerank-x")
        assert "rerank" in str(e.value)


@pytest.mark.parametrize("name", cli.EXPERIMENTS)
class TestEveryExperimentHonoursTheContract:
    """`cli` dispatches on these five names and nothing else. An experiment that
    does not carry them is not runnable, and that should fail here rather than
    half way through a four-hour run."""

    def test_it_declares_itself(self, name):
        experiment = cli.load(name)
        assert experiment.NAME == name
        assert experiment.TITLE and experiment.TITLE[0].isupper()
        assert isinstance(experiment.COSTS_MONEY, bool)

    def test_it_exposes_run_report_and_its_own_flags(self, name):
        experiment = cli.load(name)
        for attribute in ("add_run_arguments", "run", "report", "tags"):
            assert callable(getattr(experiment, attribute)), f"{name} is missing {attribute}"

    def test_its_flags_do_not_collide_with_the_standard_ones(self, name):
        """The experiment's parser is merged into `cli`'s, so a duplicate
        `--tag` would be an argparse error at import rather than a puzzle."""
        import argparse
        parser = argparse.ArgumentParser()
        cli.add_standard_arguments(parser)
        cli.load(name).add_run_arguments(parser)   # raises on a collision

    def test_it_keeps_its_output_inside_itself(self, name):
        experiment = cli.load(name)
        assert experiment.EXPERIMENT_DIR.name == name
        assert (experiment.EXPERIMENT_DIR / "README.md").exists()


class TestRunDispatch:
    def test_run_scores_then_reports_in_that_order(self, monkeypatch):
        """The whole reason there is no second verb: one command does both, and
        the report always follows the scoring."""
        called = []
        experiment = cli.load("rerank")
        monkeypatch.setattr(experiment, "run", lambda args: called.append("run"))
        monkeypatch.setattr(experiment, "report", lambda args: called.append("report"))
        monkeypatch.setattr(cli, "load", lambda name: experiment)

        cli.main(["run", "rerank", "--tag", "test"])
        assert called == ["run", "report"]

    def test_help_for_one_experiment_shows_that_experiments_flags(self, capsys):
        """`--help` after a name must show the real surface, not a union of all
        three, which is the point of merging the parsers late."""
        with pytest.raises(SystemExit):
            cli.main(["run", "rerank", "--help"])
        out = capsys.readouterr().out
        assert "--top-k" in out and "--tag" in out
        assert "--arms" not in out


def test_the_cli_runs_as_a_module_without_being_installed():
    """`python -m cli` has to work where the console script is not on PATH,
    which is every fresh container and every Kaggle kernel."""
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "cli", "list"],
                         capture_output=True, text=True, cwd=".")
    assert out.returncode == 0, out.stderr
    assert "rerank" in out.stdout


class TestAnExperimentThatIsNotCheckedOut:
    """A Kaggle kernel clones only the experiment it runs, so the others are
    genuinely absent. Listing has to survive that."""

    def test_list_skips_what_is_not_present_instead_of_crashing(self, monkeypatch, capsys):
        def absent(name):
            raise ModuleNotFoundError(f"No module named {name!r}")

        monkeypatch.setattr(cli, "load", absent)
        assert cli.main(["list"]) == 0
        out = capsys.readouterr().out
        assert "rerank" in out
        assert "not checked out" in out, "an absent experiment is stated, not hidden"


class TestStatusIsDerivedNotDeclared:
    """A hand-set status field went stale three times in one file. The results
    on disk cannot: an experiment with no results has not been run, whatever a
    constant says."""

    def test_no_results_means_it_has_not_run(self):
        assert cli.status_from_tags([]) == "not run"

    def test_a_pilot_alone_means_piloted(self):
        assert cli.status_from_tags(["pilot"]) == "piloted"

    def test_a_full_run_means_complete(self):
        assert cli.status_from_tags(["full", "pilot"]) == "complete"

    def test_any_other_tag_still_counts_as_run(self):
        """A tag named for a machine or a variant is a real measurement."""
        assert cli.status_from_tags(["gpu"]) == "complete"

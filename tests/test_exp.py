"""The CLI: what `exp` promises every experiment, and what it asks of them.

Two verbs and one contract. The tests that matter are about the contract holding
for all three experiments at once, because the point of the CLI is that a fourth
experiment costs no code here.
"""
import pytest

import exp


class TestTheRegistry:
    def test_it_names_every_experiment_in_the_repository(self):
        assert set(exp.EXPERIMENTS) == {"rerank", "banking77", "typed_decisions"}

    def test_an_unknown_name_says_what_does_exist(self):
        """Better than a traceback: the answer to 'what can I run' is the error."""
        with pytest.raises(SystemExit) as e:
            exp.load("banking-77")
        assert "banking77" in str(e.value) and "rerank" in str(e.value)


@pytest.mark.parametrize("name", exp.EXPERIMENTS)
class TestEveryExperimentHonoursTheContract:
    """`exp` dispatches on these five names and nothing else. An experiment that
    does not carry them is not runnable, and that should fail here rather than
    half way through a four-hour run."""

    def test_it_declares_itself(self, name):
        experiment = exp.load(name)
        assert experiment.NAME == name
        assert experiment.TITLE and experiment.TITLE[0].isupper()
        assert experiment.STATUS in {"designed", "piloted", "complete"}
        assert isinstance(experiment.COSTS_MONEY, bool)

    def test_it_exposes_run_report_and_its_own_flags(self, name):
        experiment = exp.load(name)
        for attribute in ("add_run_arguments", "run", "report", "tags"):
            assert callable(getattr(experiment, attribute)), f"{name} is missing {attribute}"

    def test_its_flags_do_not_collide_with_the_standard_ones(self, name):
        """The experiment's parser is merged into `exp`'s, so a duplicate
        `--tag` would be an argparse error at import rather than a puzzle."""
        import argparse
        parser = argparse.ArgumentParser()
        exp.add_standard_arguments(parser)
        exp.load(name).add_run_arguments(parser)   # raises on a collision

    def test_it_keeps_its_output_inside_itself(self, name):
        experiment = exp.load(name)
        assert experiment.EXPERIMENT_DIR.name == name
        assert (experiment.EXPERIMENT_DIR / "README.md").exists()


class TestRunDispatch:
    def test_run_scores_then_reports_in_that_order(self, monkeypatch):
        """The whole reason there is no second verb: one command does both, and
        the report always follows the scoring."""
        called = []
        experiment = exp.load("rerank")
        monkeypatch.setattr(experiment, "run", lambda args: called.append("run"))
        monkeypatch.setattr(experiment, "report", lambda args: called.append("report"))
        monkeypatch.setattr(exp, "load", lambda name: experiment)

        exp.main(["run", "rerank", "--tag", "test"])
        assert called == ["run", "report"]

    def test_help_for_one_experiment_shows_that_experiments_flags(self, capsys):
        """`--help` after a name must show the real surface, not a union of all
        three, which is the point of merging the parsers late."""
        with pytest.raises(SystemExit):
            exp.main(["run", "rerank", "--help"])
        out = capsys.readouterr().out
        assert "--top-k" in out and "--tag" in out
        assert "--arms" not in out, "that is banking77's flag, not rerank's"


class TestAFullyScoredRunNeedsNothing:
    """The property that lets one verb replace two.

    Re-running an experiment whose calls are all in the wire log must not reach
    for a key, the network or a model. If it did, recomputing a number would cost
    what measuring it costs, and there would have to be a separate report command.
    """

    def test_typed_decisions_decides_what_is_pending_from_names_alone(self):
        """Resolving a model id needs the account; knowing whether a call is
        already recorded does not. The pending check must use only the second."""
        from typed_decisions import catalog, run

        names = catalog.select("all")
        assert all(isinstance(n, str) for n in names), "select() is keyed by name"
        # arms_to_run only iterates the names and asks the static catalogue which
        # of them return a probability, so it never needs a resolved model.
        task = {"spec": {"hash": "h", "payload": {"position_bias": {"orders": []}}},
                "validation": [1]}
        planned = run.arms_to_run(task, {"main", "validation"}, names)
        assert planned, "main and validation should both be planned"
        assert all(isinstance(m, str) for _, who in planned for m in who)

    def test_it_refuses_to_ask_for_a_key_before_it_knows_there_is_work(self):
        """`main` must compute outstanding work before demanding OPENROUTER_API_KEY."""
        import inspect
        from typed_decisions import run
        source = inspect.getsource(run.main)
        assert source.index("outstanding") < source.index("no OPENROUTER_API_KEY"), (
            "the key is demanded before the pending check, so a re-run of a "
            "finished experiment would fail without one")

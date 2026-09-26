"""Submitting an experiment to Kaggle, and refusing to when it would be wrong.

The kernel clones this repository at `main`. Everything here exists because that
one sentence has consequences: work that is uncommitted, or committed but not
pushed, is simply not what runs, and the run looks completely normal while
measuring the wrong code.
"""
import pytest

from cli import kaggle


class TestReadyToSubmit:
    """Each check names what is wrong and what to do, because the alternative is
    a 40-minute run of the wrong commit."""

    def _git(self, **over):
        state = {"dirty": [], "unpushed": 0, "branch": "main"}
        state.update(over)
        return state

    def test_a_clean_pushed_checkout_is_ready(self):
        problems = kaggle.problems(self._git(), metadata=True, authenticated=True)
        assert problems == []

    def test_uncommitted_work_is_refused_and_named(self):
        problems = kaggle.problems(self._git(dirty=["rerank/evaluate.py"]),
                                   metadata=True, authenticated=True)
        assert len(problems) == 1
        assert "rerank/evaluate.py" in problems[0]
        assert "clone" in problems[0], "say why it matters, not just that it is dirty"

    def test_unpushed_commits_are_refused(self):
        problems = kaggle.problems(self._git(unpushed=3), metadata=True, authenticated=True)
        assert any("3" in p and "push" in p for p in problems)

    def test_a_branch_that_is_not_main_is_refused(self):
        """The kernel clones main. Running from another branch measures main."""
        problems = kaggle.problems(self._git(branch="spike"), metadata=True, authenticated=True)
        assert any("spike" in p and "main" in p for p in problems)

    def test_missing_credentials_are_refused(self):
        problems = kaggle.problems(self._git(), metadata=True, authenticated=False)
        assert any("kaggle" in p.lower() for p in problems)

    def test_an_experiment_with_no_kernel_is_refused(self):
        problems = kaggle.problems(self._git(), metadata=False, authenticated=True)
        assert any("kernel-metadata.json" in p for p in problems)

    def test_every_problem_is_reported_not_just_the_first(self):
        problems = kaggle.problems(self._git(dirty=["a.py"], unpushed=2, branch="spike"),
                                   metadata=False, authenticated=False)
        assert len(problems) == 5, "one round trip should surface all of them"


class TestManualPrerequisites:
    """Two things cannot be checked from here, so they are stated rather than
    silently assumed: the run fails late and confusingly otherwise."""

    def test_they_are_listed_with_where_to_do_them(self):
        notes = kaggle.manual_steps("nadeem4nk/rerank-nfcorpus-gpu")
        joined = "\n".join(notes)
        assert "OPENROUTER_API_KEY" in joined
        assert "Secrets" in joined
        assert "nadeem4nk/rerank-nfcorpus-gpu" in joined, "link to the notebook itself"
        assert any("accelerator" in n.lower() or "gpu" in n.lower() for n in notes)


class TestWhereTheKernelLives:
    def test_it_is_found_beside_the_experiment(self):
        assert kaggle.kernel_dir("rerank").name == "kaggle"
        assert kaggle.kernel_dir("rerank").parent.name == "rerank"

    def test_the_slug_comes_from_the_metadata_not_a_guess(self, tmp_path):
        (tmp_path / "kernel-metadata.json").write_text('{"id": "someone/a-kernel"}', encoding="utf-8")
        assert kaggle.slug(tmp_path) == "someone/a-kernel"

    def test_no_metadata_is_an_error_naming_the_path(self, tmp_path):
        with pytest.raises(SystemExit) as e:
            kaggle.slug(tmp_path)
        assert str(tmp_path) in str(e.value)


def test_the_dirty_list_keeps_whole_paths():
    """A fixed-width slice ate the first character of a modified path, so the
    message named a file that does not exist."""
    from cli.kaggle import git_state
    for path in git_state()["dirty"]:
        assert not path.startswith("/"), path
        head = path.split("/")[0]
        assert head in {"cli", "rerank", "banking77", "typed_decisions", "site",
                        "tests", "pyproject.toml", "uv.lock", "README.md",
                        ".gitignore", ".env.example", "EXPERIMENT_TEMPLATE.md"}, path

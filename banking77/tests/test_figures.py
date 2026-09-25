"""The figures must come out of the results file, and must refuse rather than mislead.

Two properties are worth a test. First, that `report.py` produces the images at
all, because the whole point is that nobody has to remember to draw them. Second,
that a panel whose data is missing raises instead of rendering an empty chart: a
published figure is read as a claim, and an empty one claims that nothing
happened.
"""
import pytest

from banking77 import figures


def _arm(name, accuracy, n_options=77, head_max_len=256, n_test=3080):
    return {"arm": name, "accuracy": accuracy, "n_options": n_options,
            "head_max_len": head_max_len, "n_test": n_test, "max_len": 1024,
            "checkpoint": "multilingual", "accuracy_ci95": [accuracy - 0.02, accuracy + 0.02]}


def _results(**over):
    base = {
        "tag": "test",
        "published_comparison": {"laya_banking77_accuracy": 0.425,
                                 "jev_banking77_accuracy": 0.87,
                                 "note": "published, not measured here"},
        "arms": {
            "A-default": _arm("A-default", 0.440),
            "B-384": _arm("B-384", 0.488, head_max_len=384),
            "B-512": _arm("B-512", 0.488, head_max_len=512),
            "B-768": _arm("B-768", 0.488, head_max_len=768),
            "K-10": _arm("K-10", 0.743, n_options=10, n_test=400),
            "K-20": _arm("K-20", 0.683, n_options=20, n_test=800),
            "K-40": _arm("K-40", 0.614, n_options=40, n_test=1600),
            "C-coarse-fine": _arm("C-coarse-fine", 0.344),
            "E-english": _arm("E-english", 0.365, head_max_len=192),
        },
    }
    base.update(over)
    return base


class TestWriteAll:
    def test_it_writes_every_figure_from_a_complete_result(self, tmp_path):
        out = figures.write_all(_results(), tmp_path)
        assert out["skipped"] == {}, "a complete result should skip nothing"
        assert sorted(out["written"]) == ["arms.png", "budget-cliff.png", "option-count.png"]
        for name in out["written"]:
            assert (tmp_path / name).stat().st_size > 0

    def test_a_figure_that_cannot_be_drawn_is_skipped_with_its_reason(self, tmp_path):
        arms = {k: v for k, v in _results()["arms"].items() if not k.startswith("K-")}
        out = figures.write_all(_results(arms=arms), tmp_path)
        assert "option-count" in out["skipped"]
        assert not (tmp_path / "option-count.png").exists(), "a skip must write no file"
        assert "K" in out["skipped"]["option-count"], "the reason should name what was missing"


class TestRefusals:
    """Each panel names the arms it needs, so a partial run fails loudly."""

    def test_the_budget_cliff_needs_a_raised_budget_to_compare_against(self, tmp_path):
        arms = {"A-default": _arm("A-default", 0.440)}
        with pytest.raises(figures.NothingToPlot):
            figures.budget_cliff_plot(_results(arms=arms), tmp_path)

    def test_the_option_count_panel_needs_the_sweep(self, tmp_path):
        arms = {"A-default": _arm("A-default", 0.440)}
        with pytest.raises(figures.NothingToPlot):
            figures.option_count_plot(_results(arms=arms), tmp_path)

    def test_the_arms_panel_needs_arms(self, tmp_path):
        with pytest.raises(figures.NothingToPlot):
            figures.arms_plot(_results(arms={}), tmp_path)

    def test_the_arms_panel_needs_a_full_option_arm(self, tmp_path):
        arms = {"K-10": _arm("K-10", 0.743, n_options=10, n_test=400)}
        with pytest.raises(figures.NothingToPlot):
            figures.arms_plot(_results(arms=arms), tmp_path)


class TestTheArmsPanelComparesLikeWithLike:
    """K-10 scores 0.743 against 10 options and B-768 scores 0.488 against 77.
    On one accuracy axis the shorter task looks like the better model, so the
    sweep stays out of this panel and has its own."""

    def test_it_drops_the_arms_that_change_the_option_count(self):
        shown = [a["arm"] for a in figures.comparable_arms(_results())]
        assert shown == ["C-coarse-fine", "E-english", "A-default",
                         "B-384", "B-512", "B-768"]
        assert not any(name.startswith("K-") for name in shown)


class TestTheCliffIsNotDrawnAsASlope:
    """384, 512 and 768 are identical to every decimal. A line through three
    equal points invites the reader to extrapolate a trend that is not there."""

    def test_the_raised_budget_arms_are_reported_as_one_value_when_they_agree(self):
        assert figures.budget_series(_results()) == [(256, 0.440), (384, 0.488),
                                                     (512, 0.488), (768, 0.488)]

    def test_it_says_whether_the_raised_arms_are_identical(self):
        assert figures.raised_budgets_agree(_results()) is True
        drifted = _results()
        drifted["arms"]["B-768"] = _arm("B-768", 0.501, head_max_len=768)
        assert figures.raised_budgets_agree(drifted) is False

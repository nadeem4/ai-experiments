"""Percentiles, a bootstrap CI, agreement, and the spend estimate that guards a run."""
import pytest

from typed_decisions import stats


def test_p50_of_an_odd_sample_is_the_middle_value():
    assert stats.percentile([3, 1, 2], 50) == 2


def test_p95_is_nearest_rank_so_it_is_always_an_observed_latency():
    """Interpolating would report a latency no call actually had. Nearest-rank
    reports one that did."""
    assert stats.percentile(list(range(1, 101)), 95) == 95
    assert stats.percentile([10, 20], 95) == 20


def test_percentile_of_nothing_is_none_rather_than_zero():
    assert stats.percentile([], 50) is None


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    values = [1, 0, 1, 1, 0, 1, 1, 1, 0, 1]
    low, high = stats.bootstrap_ci(values, seed=0)
    assert low <= sum(values) / len(values) <= high
    assert stats.bootstrap_ci(values, seed=0) == (low, high)


def test_bootstrap_ci_of_a_unanimous_sample_is_a_point():
    assert stats.bootstrap_ci([1, 1, 1], seed=0) == (1.0, 1.0)










def test_spend_estimate_uses_measured_tokens_and_the_catalogs_price():
    per_call = stats.call_cost(input_tokens=110, output_tokens=7,
                               pricing={"input": "0.000001", "output": "0.00001"})
    assert per_call == pytest.approx(110e-6 + 70e-6)


def test_spend_estimate_scales_by_calls_including_the_warm_ups():
    plan = stats.estimate([
        {"model": "gpt", "calls": 100, "warmup": 20, "cost_per_call": 0.001},
        {"model": "jev", "calls": 100, "warmup": 20, "cost_per_call": 0.00002},
    ])
    assert plan["total_usd"] == pytest.approx(120 * 0.001 + 120 * 0.00002)
    assert plan["by_model"]["gpt"] == pytest.approx(0.12)


def test_a_run_under_the_threshold_needs_no_confirmation():
    assert stats.needs_confirmation(0.4, threshold=1.0) is False


def test_a_run_over_the_threshold_needs_the_explicit_flag():
    assert stats.needs_confirmation(1.01, threshold=1.0) is True


# --- the three metrics added with the OpenRouter migration -------------------


def test_cost_per_correct_decision_divides_cost_by_accuracy():
    """The headline number. Dollars per call is misleading when accuracy differs:
    a model that is half the price and half as accurate costs the same per decision
    you can actually use."""
    assert stats.cost_per_correct_decision(0.001, 0.5) == pytest.approx(0.002)
    assert stats.cost_per_correct_decision(0.001, 1.0) == pytest.approx(0.001)


def test_cost_per_correct_decision_is_undefined_at_zero_accuracy_not_infinite():
    """A model that is never right has no cost per correct decision. Dividing by
    zero would print `inf` and sort to the bottom of a table as though it were a
    number someone measured."""
    assert stats.cost_per_correct_decision(0.001, 0.0) is None


def test_cost_per_correct_decision_is_undefined_where_there_is_no_gold():
    assert stats.cost_per_correct_decision(0.001, None) is None
    assert stats.cost_per_correct_decision(None, 0.5) is None


def test_tail_ratio_is_p95_over_p50():
    """689 ms p50 against 7,093 ms p95 is a 10x tail that a p50-only table hides
    entirely, and a 10x tail is worse in production than a slower steady model."""
    assert stats.tail_ratio(p95=7093.0, p50=689.0) == pytest.approx(10.29, rel=1e-3)


def test_tail_ratio_of_a_perfectly_steady_model_is_one():
    assert stats.tail_ratio(p95=100.0, p50=100.0) == pytest.approx(1.0)


def test_tail_ratio_is_none_rather_than_a_division_by_zero():
    assert stats.tail_ratio(p95=100.0, p50=None) is None
    assert stats.tail_ratio(p95=None, p50=100.0) is None
    assert stats.tail_ratio(p95=100.0, p50=0.0) is None












# --- position bias -----------------------------------------------------------
#
# Same example, same options, only the order changes, so any change of answer is
# pure error. The flip rate says how much a model moves; only the controlled gold
# placement says which positions it favours.


class TestFlipRate:
    def test_a_model_that_never_changes_its_answer_has_a_flip_rate_of_zero(self):
        answers = {"a": ["World", "World", "World"], "b": ["Sports", "Sports", "Sports"]}
        assert stats.flip_rate(answers) == {"rate": 0.0, "n": 2, "n_incomplete": 0}

    def test_a_model_that_always_changes_its_answer_has_a_flip_rate_of_one(self):
        answers = {"a": ["World", "Sports", "Business"]}
        assert stats.flip_rate(answers)["rate"] == 1.0

    def test_one_disagreeing_order_out_of_three_still_counts_as_a_flip(self):
        """The metric is 'the answer is not the same across all three', not a
        pairwise average: two out of three agreeing is still an error."""
        answers = {"a": ["World", "World", "Sports"], "b": ["Sports"] * 3}
        assert stats.flip_rate(answers)["rate"] == 0.5

    def test_an_example_missing_an_order_is_excluded_and_counted(self):
        """A model that failed one of the three calls has not demonstrated
        stability over that example, and it has not demonstrated a flip either.
        It is reported as excluded rather than silently helping or hurting."""
        answers = {"a": ["World", "World"], "b": ["Sports"] * 3}
        assert stats.flip_rate(answers, n_orders=3) == {"rate": 0.0, "n": 1, "n_incomplete": 1}

    def test_an_unusable_answer_is_not_evidence_of_stability(self):
        """Three Nones is three failures, not a model agreeing with itself."""
        answers = {"a": [None, None, None]}
        assert stats.flip_rate(answers)["rate"] == 1.0

    def test_no_examples_gives_no_rate_rather_than_zero(self):
        assert stats.flip_rate({})["rate"] is None


class TestAccuracyByGoldPosition:
    def test_accuracy_is_reported_at_each_placement_and_the_spread_between_them(self):
        by_placement = {"first": [1, 1, 1, 1], "middle": [1, 1, 0, 0], "last": [1, 0, 0, 0]}
        out = stats.accuracy_by_position(by_placement)
        assert out["accuracy"] == {"first": 1.0, "middle": 0.5, "last": 0.25}
        assert out["spread"] == 0.75
        assert out["n"] == {"first": 4, "middle": 4, "last": 4}

    def test_a_model_with_no_position_preference_has_a_spread_of_zero(self):
        by_placement = {"first": [1, 0], "middle": [1, 0], "last": [0, 1]}
        assert stats.accuracy_by_position(by_placement)["spread"] == 0.0

    def test_the_spread_needs_every_placement_to_be_measured(self):
        """Two placements out of three is not a spread, and reporting one would
        be a comparison the run never made."""
        assert stats.accuracy_by_position({"first": [1], "last": [0]})["spread"] is None

    def test_an_empty_placement_is_not_scored_as_zero_accuracy(self):
        out = stats.accuracy_by_position({"first": [1], "middle": [], "last": [1]})
        assert out["accuracy"]["middle"] is None
        assert out["spread"] is None


class TestChosenPositionDistribution:
    def test_positions_are_read_off_the_order_that_was_actually_shown(self):
        shown = ["Sports", "World", "Business"]
        assert stats.chosen_position("World", shown) == 1
        assert stats.chosen_position("Sports", shown) == 0

    def test_an_answer_that_was_not_on_the_list_has_no_position(self):
        assert stats.chosen_position("Nonsense", ["a", "b"]) is None
        assert stats.chosen_position(None, ["a", "b"]) is None

    def test_a_four_option_histogram_is_one_bin_per_option(self):
        out = stats.position_histogram([0, 0, 1, 3], n_options=4)
        assert out["shares"] == [0.5, 0.25, 0.0, 0.25]
        assert out["n"] == 4

    def test_a_long_option_list_is_binned_so_the_shape_is_readable(self):
        out = stats.position_histogram(list(range(151)), n_options=151, n_bins=10)
        assert len(out["shares"]) == 10
        assert abs(sum(out["shares"]) - 1.0) < 1e-9

    def test_the_mean_normalised_position_says_which_end_a_model_leans_to(self):
        """0 is 'always picks the first option', 1 is 'always picks the last'.
        A model with no positional bias sits near the middle."""
        assert stats.position_histogram([0, 0, 0], n_options=4)["mean_normalised"] == 0.0
        assert stats.position_histogram([3, 3, 3], n_options=4)["mean_normalised"] == 1.0
        assert stats.position_histogram([0, 3], n_options=4)["mean_normalised"] == 0.5

    def test_no_positions_gives_no_distribution_rather_than_a_flat_one(self):
        out = stats.position_histogram([], n_options=4)
        assert out["shares"] is None and out["mean_normalised"] is None and out["n"] == 0


# --- paired comparison -------------------------------------------------------


class TestPairedDifference:
    def test_it_is_computed_over_the_examples_both_models_answered(self):
        a = {"x": 1, "y": 1, "z": 0}
        b = {"x": 0, "y": 1, "w": 1}
        out = stats.paired(a, b)
        assert out["n"] == 2
        assert out["mean_diff"] == 0.5

    def test_the_count_is_reported_because_the_intersection_can_be_small(self):
        assert stats.paired({"x": 1}, {"y": 0})["n"] == 0
        assert stats.paired({"x": 1}, {"y": 0})["mean_diff"] is None

    def test_the_interval_is_paired_rather_than_two_overlapping_intervals(self):
        """Per-item differences, because shared per-item variation swamps the
        effect otherwise and two overlapping intervals say nothing."""
        a = {f"e{i}": 1 for i in range(40)}
        b = {f"e{i}": 0 for i in range(40)}
        out = stats.paired(a, b)
        assert out["ci"][0] > 0

    def test_a_difference_of_zero_has_an_interval_containing_zero(self):
        a = {f"e{i}": i % 2 for i in range(40)}
        out = stats.paired(a, dict(a))
        assert out["mean_diff"] == 0.0
        assert out["ci"][0] <= 0 <= out["ci"][1]

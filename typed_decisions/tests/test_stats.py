"""Percentiles, a bootstrap CI, agreement, and the spend estimate that guards a run."""
import pytest

import math

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
        assert stats.flip_rate(answers) == {"rate": 0.0, "n": 2, "n_incomplete": 0,
                                            "n_unusable": 0, "n_failed_call": 0}

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
        assert stats.flip_rate(answers, n_orders=3) == {"rate": 0.0, "n": 1, "n_incomplete": 1,
                                                        "n_unusable": 0, "n_failed_call": 0}

    def test_an_example_the_model_never_answered_is_excluded_and_counted(self):
        """Three Nones is three failures. It is not a model agreeing with itself,
        and it is not position bias either -- attributing a total failure to the
        option order would put a number in the bias table that the order did not
        cause. Laya on a 151-option list does exactly this."""
        answers = {"a": [None, None, None], "b": ["World"] * 3}
        assert stats.flip_rate(answers) == {"rate": 0.0, "n": 1, "n_incomplete": 0,
                                            "n_unusable": 1, "n_failed_call": 0}

    def test_an_example_whose_call_failed_is_excluded_not_counted_as_a_flip(self):
        """A null from a failed call is not evidence that the ORDER changed the
        answer, which is the only thing this rate measures. Excluded and counted,
        the same treatment an all-failed example already gets, and for the same
        reason."""
        answers = {"a": ["World", "World", None], "b": ["Sports"] * 3}
        assert stats.flip_rate(answers) == {"rate": 0.0, "n": 1, "n_incomplete": 0,
                                            "n_unusable": 0, "n_failed_call": 1}

    def test_a_failed_call_does_not_hide_a_real_flip_elsewhere(self):
        answers = {"a": ["World", "Sports", "World"], "b": ["Sports", "Sports", None]}
        out = stats.flip_rate(answers)
        assert out["rate"] == 1.0 and out["n"] == 1 and out["n_failed_call"] == 1

    def test_a_model_that_answered_nothing_at_all_has_no_flip_rate(self):
        assert stats.flip_rate({"a": [None] * 3})["rate"] is None

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


# =============================================================================
# Calibration, for the models that return a probability
# =============================================================================
#
# Calibration, for the models that return a probability at all.
#
# Only Jev and Laya do. An LLM in structured mode returns a label, so there is
# nothing to calibrate and nothing to threshold -- which is the point of reporting
# the capability as a column rather than a footnote.
#
# The rule these tests exist to hold: **nothing is ever fitted on test.** The
# temperature is fitted on a validation split carved out of train.


def confident(p, gold="a"):
    """A two-option distribution with probability `p` on option 'a'."""
    return {"a": p, "b": 1 - p}, gold


class TestECE:
    def test_a_perfectly_calibrated_model_has_an_ece_near_zero(self):
        """Half the 0.5-confidence answers right, all the 1.0 ones right."""
        confidences = [0.5] * 100 + [1.0] * 100
        correct = [1, 0] * 50 + [1] * 100
        assert stats.ece(confidences, correct, n_bins=15) < 0.01

    def test_an_overconfident_model_has_a_large_ece(self):
        confidences = [0.99] * 100
        correct = [1] * 50 + [0] * 50
        assert stats.ece(confidences, correct, n_bins=15) == pytest.approx(0.49, abs=0.01)

    def test_an_underconfident_model_is_also_miscalibrated(self):
        """ECE is a distance, so being too humble counts too."""
        confidences = [0.5] * 100
        correct = [1] * 100
        assert stats.ece(confidences, correct, n_bins=15) == pytest.approx(0.5, abs=0.01)

    def test_it_uses_fifteen_bins_by_default(self):
        assert stats.N_BINS == 15

    def test_empty_input_gives_no_ece_rather_than_zero(self):
        assert stats.ece([], [], n_bins=15) is None

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            stats.ece([0.5, 0.5], [1], n_bins=15)

    def test_bins_with_nothing_in_them_do_not_drag_the_score_down(self):
        """An empty bin is not evidence of calibration; it is weighted by the
        share of samples in it, which is zero."""
        assert stats.ece([1.0] * 10, [1] * 10, n_bins=15) == pytest.approx(0.0, abs=1e-9)


class TestTemperature:
    def test_an_overconfident_model_is_fitted_a_temperature_above_one(self):
        """T > 1 flattens the distribution, which is what an overconfident model
        needs."""
        rows = [confident(0.99, "a") for _ in range(50)] + [confident(0.99, "b") for _ in range(50)]
        assert stats.fit_temperature(rows) > 1.0

    def test_an_underconfident_model_is_fitted_a_temperature_below_one(self):
        rows = [confident(0.55, "a") for _ in range(95)] + [confident(0.55, "b") for _ in range(5)]
        assert stats.fit_temperature(rows) < 1.0

    def test_a_well_calibrated_model_is_fitted_a_temperature_near_one(self):
        rows = [confident(0.8, "a") for _ in range(80)] + [confident(0.8, "b") for _ in range(20)]
        assert stats.fit_temperature(rows) == pytest.approx(1.0, abs=0.25)

    def test_applying_a_temperature_of_one_changes_nothing(self):
        probs = {"a": 0.7, "b": 0.2, "c": 0.1}
        out = stats.apply_temperature(probs, 1.0)
        for k in probs:
            assert out[k] == pytest.approx(probs[k])

    def test_a_high_temperature_flattens_the_distribution(self):
        out = stats.apply_temperature({"a": 0.9, "b": 0.1}, 5.0)
        assert out["a"] < 0.9
        assert sum(out.values()) == pytest.approx(1.0)

    def test_a_low_temperature_sharpens_it(self):
        out = stats.apply_temperature({"a": 0.6, "b": 0.4}, 0.2)
        assert out["a"] > 0.6

    def test_the_rescaled_distribution_still_sums_to_one(self):
        out = stats.apply_temperature({"a": 0.5, "b": 0.3, "c": 0.2}, 2.7)
        assert sum(out.values()) == pytest.approx(1.0)

    def test_the_argmax_is_never_moved_by_a_temperature(self):
        """Temperature scaling is a calibration fix, not an accuracy fix. If the
        accuracy moved, something else changed."""
        probs = {"a": 0.5, "b": 0.3, "c": 0.2}
        for t in (0.3, 1.0, 4.0):
            assert max(stats.apply_temperature(probs, t), key=lambda k: probs[k]) == "a"

    def test_a_zero_probability_does_not_blow_up_the_fit(self):
        """A hard 0 on an option is ordinary; log(0) would end the fit rather
        than inform it."""
        rows = [({"a": 1.0, "b": 0.0}, "a") for _ in range(stats.MIN_VALIDATION)]
        assert math.isfinite(stats.fit_temperature(rows))

    def test_nothing_to_fit_on_gives_no_temperature_rather_than_one(self):
        """Silently returning 1.0 would report an unfitted model as calibrated."""
        assert stats.fit_temperature([]) is None

    def test_a_handful_of_validation_rows_is_refused_rather_than_fitted(self):
        """With four rows that the model got right and was confident about, the
        likelihood is minimised by driving T towards zero -- the fit reports a
        temperature of 0.02 and an ECE of exactly zero, which is not a
        calibration result, it is an artefact of the sample size. Below the
        minimum, no temperature is reported at all."""
        rows = [confident(0.99, "a") for _ in range(4)]
        assert stats.fit_temperature(rows) is None
        assert stats.MIN_VALIDATION > 4

    def test_at_the_minimum_it_fits_again(self):
        rows = ([confident(0.99, "a") for _ in range(stats.MIN_VALIDATION // 2)]
                + [confident(0.99, "b") for _ in range(stats.MIN_VALIDATION // 2)])
        assert stats.fit_temperature(rows) is not None

    def test_the_fitted_temperature_stays_inside_a_sane_range(self):
        """A T of 0.001 is not a calibration, it is the optimiser running off the
        end of the grid."""
        rows = [confident(0.999, "a") for _ in range(50)]
        fitted = stats.fit_temperature(rows)
        assert stats.T_MIN <= fitted <= stats.T_MAX


class TestTheReportedPair:
    def test_raw_and_scaled_ece_are_both_returned_with_the_temperature(self):
        val = [confident(0.99, "a") for _ in range(50)] + [confident(0.99, "b") for _ in range(50)]
        test = [confident(0.99, "a") for _ in range(50)] + [confident(0.99, "b") for _ in range(50)]
        out = stats.calibrate(validation=val, test=test)
        assert out["temperature"] > 1
        assert out["ece_raw"] > out["ece_scaled"]
        assert out["n_validation"] == 100 and out["n_test"] == 100

    def test_the_temperature_is_fitted_on_validation_and_never_on_test(self):
        """The fit must not see the test rows. Passing a different test set must
        not change the temperature."""
        val = [confident(0.99, "a") for _ in range(50)] + [confident(0.99, "b") for _ in range(50)]
        one = stats.calibrate(validation=val, test=[confident(0.9, "a")])
        two = stats.calibrate(validation=val, test=[confident(0.1, "b")] * 30)
        assert one["temperature"] == two["temperature"]

    def test_too_few_validation_rows_gives_a_raw_ece_and_no_scaled_one(self):
        """Better than publishing a temperature fitted on four rows."""
        out = stats.calibrate(validation=[confident(0.9, "a")] * 4,
                                    test=[confident(0.9, "a")] * 10)
        assert out["ece_raw"] is not None
        assert out["temperature"] is None and out["ece_scaled"] is None

    def test_a_model_with_no_probabilities_is_absent_rather_than_scored(self):
        assert stats.calibrate(validation=[], test=[]) is None


class TestReliabilityCurve:
    def test_it_returns_one_point_per_non_empty_bin(self):
        out = stats.reliability([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1], n_bins=10)
        assert out["bin_centres"] == pytest.approx([0.05, 0.85], abs=0.06)
        assert out["accuracy"] == [0.0, 1.0]
        assert out["counts"] == [2, 2]

    def test_empty_bins_are_left_out_rather_than_drawn_at_zero(self):
        """A bin nothing fell into is not a model that was wrong there."""
        out = stats.reliability([0.95] * 5, [1] * 5, n_bins=15)
        assert len(out["accuracy"]) == 1

    def test_the_bin_centre_is_the_mean_confidence_in_the_bin(self):
        """Plotted against the diagonal, so the x has to be what the model
        actually said rather than the nominal centre of the bucket."""
        out = stats.reliability([0.80, 0.86], [1, 1], n_bins=10)
        assert out["bin_centres"][0] == pytest.approx(0.83)

    def test_nothing_to_plot_gives_no_curve(self):
        assert stats.reliability([], [], n_bins=15) is None


class TestMcNemar:
    """The paired test the gold-placement arm needs.

    Two placements are run over the same examples, so the question is not whether
    two rates differ but whether the examples that changed changed in one
    direction. Only the discordant pairs carry information; the exact binomial
    over them is what the 40-example subset can support.
    """

    def test_no_discordant_pairs_cannot_show_an_effect(self):
        out = stats.mcnemar([1, 1, 0, 0], [1, 1, 0, 0])
        assert out == {"n01": 0, "n10": 0, "n_discordant": 0, "p_value": 1.0}

    def test_a_lopsided_split_is_significant_and_a_balanced_one_is_not(self):
        lopsided = stats.mcnemar([1] * 6 + [0] * 6, [0] * 6 + [0] * 6)
        assert lopsided["n_discordant"] == 6
        assert lopsided["p_value"] == pytest.approx(0.03125), "six one-sided pairs clear 0.05"

        balanced = stats.mcnemar([1, 0, 1, 0], [0, 1, 0, 1])
        assert balanced["n_discordant"] == 4
        assert balanced["p_value"] == 1.0

    def test_it_reproduces_the_run_s_one_significant_result(self):
        """phi/clinc150: 11 examples right only when the gold option was first,
        2 right only when it was last. The single effect the run detected."""
        a = [1] * 11 + [0] * 2
        b = [0] * 11 + [1] * 2
        out = stats.mcnemar(a, b)
        assert out["n_discordant"] == 13
        assert out["p_value"] == pytest.approx(0.02246, abs=1e-5)

    def test_it_refuses_vectors_of_different_lengths(self):
        with pytest.raises(ValueError):
            stats.mcnemar([1, 0], [1, 0, 1])

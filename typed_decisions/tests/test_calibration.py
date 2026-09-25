"""Calibration, for the models that return a probability at all.

Only Jev and Laya do. An LLM in structured mode returns a label, so there is
nothing to calibrate and nothing to threshold -- which is the point of reporting
the capability as a column rather than a footnote.

The rule these tests exist to hold: **nothing is ever fitted on test.** The
temperature is fitted on a validation split carved out of train.
"""
import math

import pytest

from typed_decisions import calibration


def confident(p, gold="a"):
    """A two-option distribution with probability `p` on option 'a'."""
    return {"a": p, "b": 1 - p}, gold


class TestECE:
    def test_a_perfectly_calibrated_model_has_an_ece_near_zero(self):
        """Half the 0.5-confidence answers right, all the 1.0 ones right."""
        confidences = [0.5] * 100 + [1.0] * 100
        correct = [1, 0] * 50 + [1] * 100
        assert calibration.ece(confidences, correct, n_bins=15) < 0.01

    def test_an_overconfident_model_has_a_large_ece(self):
        confidences = [0.99] * 100
        correct = [1] * 50 + [0] * 50
        assert calibration.ece(confidences, correct, n_bins=15) == pytest.approx(0.49, abs=0.01)

    def test_an_underconfident_model_is_also_miscalibrated(self):
        """ECE is a distance, so being too humble counts too."""
        confidences = [0.5] * 100
        correct = [1] * 100
        assert calibration.ece(confidences, correct, n_bins=15) == pytest.approx(0.5, abs=0.01)

    def test_it_uses_fifteen_bins_by_default(self):
        assert calibration.N_BINS == 15

    def test_empty_input_gives_no_ece_rather_than_zero(self):
        assert calibration.ece([], [], n_bins=15) is None

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError):
            calibration.ece([0.5, 0.5], [1], n_bins=15)

    def test_bins_with_nothing_in_them_do_not_drag_the_score_down(self):
        """An empty bin is not evidence of calibration; it is weighted by the
        share of samples in it, which is zero."""
        assert calibration.ece([1.0] * 10, [1] * 10, n_bins=15) == pytest.approx(0.0, abs=1e-9)


class TestTemperature:
    def test_an_overconfident_model_is_fitted_a_temperature_above_one(self):
        """T > 1 flattens the distribution, which is what an overconfident model
        needs."""
        rows = [confident(0.99, "a") for _ in range(50)] + [confident(0.99, "b") for _ in range(50)]
        assert calibration.fit_temperature(rows) > 1.0

    def test_an_underconfident_model_is_fitted_a_temperature_below_one(self):
        rows = [confident(0.55, "a") for _ in range(95)] + [confident(0.55, "b") for _ in range(5)]
        assert calibration.fit_temperature(rows) < 1.0

    def test_a_well_calibrated_model_is_fitted_a_temperature_near_one(self):
        rows = [confident(0.8, "a") for _ in range(80)] + [confident(0.8, "b") for _ in range(20)]
        assert calibration.fit_temperature(rows) == pytest.approx(1.0, abs=0.25)

    def test_applying_a_temperature_of_one_changes_nothing(self):
        probs = {"a": 0.7, "b": 0.2, "c": 0.1}
        out = calibration.apply_temperature(probs, 1.0)
        for k in probs:
            assert out[k] == pytest.approx(probs[k])

    def test_a_high_temperature_flattens_the_distribution(self):
        out = calibration.apply_temperature({"a": 0.9, "b": 0.1}, 5.0)
        assert out["a"] < 0.9
        assert sum(out.values()) == pytest.approx(1.0)

    def test_a_low_temperature_sharpens_it(self):
        out = calibration.apply_temperature({"a": 0.6, "b": 0.4}, 0.2)
        assert out["a"] > 0.6

    def test_the_rescaled_distribution_still_sums_to_one(self):
        out = calibration.apply_temperature({"a": 0.5, "b": 0.3, "c": 0.2}, 2.7)
        assert sum(out.values()) == pytest.approx(1.0)

    def test_the_argmax_is_never_moved_by_a_temperature(self):
        """Temperature scaling is a calibration fix, not an accuracy fix. If the
        accuracy moved, something else changed."""
        probs = {"a": 0.5, "b": 0.3, "c": 0.2}
        for t in (0.3, 1.0, 4.0):
            assert max(calibration.apply_temperature(probs, t), key=lambda k: probs[k]) == "a"

    def test_a_zero_probability_does_not_blow_up_the_fit(self):
        rows = [({"a": 1.0, "b": 0.0}, "a") for _ in range(10)]
        assert math.isfinite(calibration.fit_temperature(rows))

    def test_nothing_to_fit_on_gives_no_temperature_rather_than_one(self):
        """Silently returning 1.0 would report an unfitted model as calibrated."""
        assert calibration.fit_temperature([]) is None


class TestTheReportedPair:
    def test_raw_and_scaled_ece_are_both_returned_with_the_temperature(self):
        val = [confident(0.99, "a") for _ in range(50)] + [confident(0.99, "b") for _ in range(50)]
        test = [confident(0.99, "a") for _ in range(50)] + [confident(0.99, "b") for _ in range(50)]
        out = calibration.calibrate(validation=val, test=test)
        assert out["temperature"] > 1
        assert out["ece_raw"] > out["ece_scaled"]
        assert out["n_validation"] == 100 and out["n_test"] == 100

    def test_the_temperature_is_fitted_on_validation_and_never_on_test(self):
        """The fit must not see the test rows. Passing a different test set must
        not change the temperature."""
        val = [confident(0.99, "a") for _ in range(50)] + [confident(0.99, "b") for _ in range(50)]
        one = calibration.calibrate(validation=val, test=[confident(0.9, "a")])
        two = calibration.calibrate(validation=val, test=[confident(0.1, "b")] * 30)
        assert one["temperature"] == two["temperature"]

    def test_a_model_with_no_probabilities_is_absent_rather_than_scored(self):
        assert calibration.calibrate(validation=[], test=[]) is None


class TestReliabilityCurve:
    def test_it_returns_one_point_per_non_empty_bin(self):
        out = calibration.reliability([0.1, 0.1, 0.9, 0.9], [0, 0, 1, 1], n_bins=10)
        assert out["bin_centres"] == pytest.approx([0.05, 0.85], abs=0.06)
        assert out["accuracy"] == [0.0, 1.0]
        assert out["counts"] == [2, 2]

    def test_empty_bins_are_left_out_rather_than_drawn_at_zero(self):
        """A bin nothing fell into is not a model that was wrong there."""
        out = calibration.reliability([0.95] * 5, [1] * 5, n_bins=15)
        assert len(out["accuracy"]) == 1

    def test_the_bin_centre_is_the_mean_confidence_in_the_bin(self):
        """Plotted against the diagonal, so the x has to be what the model
        actually said rather than the nominal centre of the bucket."""
        out = calibration.reliability([0.80, 0.86], [1, 1], n_bins=10)
        assert out["bin_centres"][0] == pytest.approx(0.83)

    def test_nothing_to_plot_gives_no_curve(self):
        assert calibration.reliability([], [], n_bins=15) is None

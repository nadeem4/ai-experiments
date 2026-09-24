"""Percentiles, a bootstrap CI, agreement, and the spend estimate that guards a run."""
import pytest

from latency import stats


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


def test_agreement_is_the_share_of_shared_examples_two_models_answer_the_same_way():
    a = {"e1": "IDLE", "e2": "FASTER", "e3": "IDLE"}
    b = {"e1": "IDLE", "e2": "SLOWER", "e3": "IDLE"}
    assert stats.agreement(a, b) == (2 / 3, 3)


def test_agreement_only_counts_examples_both_models_answered():
    a = {"e1": "IDLE", "e2": "FASTER"}
    b = {"e1": "IDLE", "e3": "IDLE"}
    assert stats.agreement(a, b) == (1.0, 1)


def test_agreement_with_no_shared_examples_is_none():
    assert stats.agreement({"e1": "IDLE"}, {"e2": "IDLE"}) == (None, 0)


def test_an_invalid_answer_is_not_agreement():
    """A model that returned nothing usable agrees with nobody, including another
    model that also returned nothing usable."""
    assert stats.agreement({"e1": None}, {"e1": None}) == (0.0, 1)


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

"""Turning the call log into the per-model row the report prints."""
import pytest

from latency import report


def _call(model="gpt", latency=100.0, validity="valid", choice="IDLE", **kw):
    base = {"model": model, "model_id": f"vendor/{model}", "task": "highway", "example_id": "e",
            "latency_ms": latency, "wall_ms": latency, "input_tokens": 110, "output_tokens": 7,
            "market_cost": 1e-05, "validity": validity, "choice": choice, "correct": None,
            "retries": 0, "failed": False, "structured": True, "n_options": 5}
    return {**base, **kw}


def test_latency_percentiles_use_only_the_calls_that_succeeded():
    """A failed call has no latency to report -- None, not zero -- and a zero would
    drag the median down and make a broken model look fast."""
    rows = [_call(latency=10.0), _call(latency=20.0), _call(latency=30.0),
            _call(latency=None, validity="api_error", failed=True)]
    summary = report.summarise(rows)["gpt/highway"]
    assert summary["p50_ms"] == 20.0 and summary["n_timed"] == 3


def test_validity_is_counted_by_kind_and_totals_the_calls():
    rows = [_call(), _call(validity="unparseable", choice=None),
            _call(validity="not_an_option", choice=None), _call()]
    summary = report.summarise(rows)["gpt/highway"]
    assert summary["valid"] == 2
    assert summary["validity"] == {"valid": 2, "unparseable": 1, "not_an_option": 1}
    assert summary["valid_rate"] == 0.5
    assert summary["n"] == 4


def test_cost_is_summed_from_the_gateways_own_per_call_figure():
    rows = [_call(market_cost=1e-05), _call(market_cost=3e-05)]
    summary = report.summarise(rows)["gpt/highway"]
    assert summary["cost_usd"] == pytest.approx(4e-05)
    assert summary["cost_per_1k_usd"] == pytest.approx(0.02)


def test_tokens_are_averaged_only_over_the_calls_that_reported_them():
    rows = [_call(input_tokens=100), _call(input_tokens=None, output_tokens=None)]
    summary = report.summarise(rows)["gpt/highway"]
    assert summary["mean_input_tokens"] == 100
    assert summary["n_tokens_reported"] == 1


def test_a_model_that_never_reports_tokens_says_so_rather_than_estimating():
    rows = [_call(model="laya", input_tokens=None, output_tokens=None)]
    summary = report.summarise(rows)["laya/highway"]
    assert summary["mean_input_tokens"] is None


def test_accuracy_is_reported_with_a_bootstrap_interval_where_there_is_gold():
    rows = [_call(correct=1) for _ in range(8)] + [_call(correct=0) for _ in range(2)]
    summary = report.summarise(rows)["gpt/highway"]
    assert summary["accuracy"] == pytest.approx(0.8)
    low, high = summary["accuracy_ci"]
    assert low <= 0.8 <= high


def test_a_task_with_no_gold_reports_no_accuracy_at_all():
    summary = report.summarise([_call(correct=None)])["gpt/highway"]
    assert summary["accuracy"] is None and summary["accuracy_ci"] is None


def test_an_invalid_answer_counts_as_wrong_not_as_missing():
    """Dropping unparseable answers before scoring would pay a model for failing."""
    rows = [_call(correct=1), _call(validity="unparseable", choice=None, correct=0)]
    assert report.summarise(rows)["gpt/highway"]["accuracy"] == pytest.approx(0.5)


def test_retries_and_failures_are_carried_through():
    rows = [_call(retries=2), _call(retries=0, failed=True, validity="api_error")]
    summary = report.summarise(rows)["gpt/highway"]
    assert summary["retries"] == 2 and summary["failed"] == 1


def test_choices_by_example_feed_the_agreement_matrix():
    rows = [_call(example_id="e1"), _call(example_id="e2", validity="unparseable", choice=None)]
    assert report.choices(rows, "gpt", "highway") == {"e1": "IDLE", "e2": None}

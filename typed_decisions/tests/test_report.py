"""Turning the call log into the per-model row the report prints."""
import pytest

from typed_decisions import report

KEY = "gpt/highway@openrouter"


def _call(model="gpt", latency=100.0, validity="valid", choice="IDLE", **kw):
    base = {"model": model, "model_id": f"vendor/{model}", "task": "highway", "example_id": "e",
            "latency_ms": latency, "wall_ms": latency, "input_tokens": 110, "output_tokens": 7,
            "cost": 1e-05, "validity": validity, "choice": choice, "correct": None,
            "retries": 0, "failed": False, "structured": True, "n_options": 5,
            "transport": "openrouter", "repeat": 0}
    return {**base, **kw}


def test_latency_percentiles_use_only_the_calls_that_succeeded():
    """A failed call has no latency to report -- None, not zero -- and a zero would
    drag the median down and make a broken model look fast."""
    rows = [_call(latency=10.0), _call(latency=20.0), _call(latency=30.0),
            _call(latency=None, validity="api_error", failed=True)]
    summary = report.summarise(rows)[KEY]
    assert summary["p50_ms"] == 20.0 and summary["n_timed"] == 3


def test_validity_is_counted_by_kind_and_totals_the_calls():
    rows = [_call(), _call(validity="unparseable", choice=None),
            _call(validity="not_an_option", choice=None), _call()]
    summary = report.summarise(rows)[KEY]
    assert summary["valid"] == 2
    assert summary["validity"] == {"valid": 2, "unparseable": 1, "not_an_option": 1}
    assert summary["valid_rate"] == 0.5
    assert summary["n"] == 4


def test_cost_is_summed_from_the_providers_own_per_call_figure():
    rows = [_call(cost=1e-05), _call(cost=3e-05)]
    summary = report.summarise(rows)[KEY]
    assert summary["cost_usd"] == pytest.approx(4e-05)
    assert summary["cost_per_1k_usd"] == pytest.approx(0.02)


def test_tokens_are_averaged_only_over_the_calls_that_reported_them():
    rows = [_call(input_tokens=100), _call(input_tokens=None, output_tokens=None)]
    summary = report.summarise(rows)[KEY]
    assert summary["mean_input_tokens"] == 100
    assert summary["n_tokens_reported"] == 1


def test_a_model_that_never_reports_tokens_says_so_rather_than_estimating():
    rows = [_call(model="laya", input_tokens=None, output_tokens=None, transport="local-cpu")]
    summary = report.summarise(rows)["laya/highway@local-cpu"]
    assert summary["mean_input_tokens"] is None


def test_accuracy_is_reported_with_a_bootstrap_interval_where_there_is_gold():
    rows = [_call(correct=1) for _ in range(8)] + [_call(correct=0) for _ in range(2)]
    summary = report.summarise(rows)[KEY]
    assert summary["accuracy"] == pytest.approx(0.8)
    low, high = summary["accuracy_ci"]
    assert low <= 0.8 <= high


def test_a_task_with_no_gold_reports_no_accuracy_at_all():
    summary = report.summarise([_call(correct=None)])[KEY]
    assert summary["accuracy"] is None and summary["accuracy_ci"] is None


def test_an_invalid_answer_counts_as_wrong_not_as_missing():
    """Dropping unparseable answers before scoring would pay a model for failing."""
    rows = [_call(correct=1), _call(validity="unparseable", choice=None, correct=0)]
    assert report.summarise(rows)[KEY]["accuracy"] == pytest.approx(0.5)


def test_retries_and_failures_are_carried_through():
    rows = [_call(retries=2), _call(retries=0, failed=True, validity="api_error")]
    summary = report.summarise(rows)[KEY]
    assert summary["retries"] == 2 and summary["failed"] == 1


# --- the transport tag -------------------------------------------------------


def test_two_transports_are_never_averaged_into_one_row():
    """The experiment moved from the Vercel gateway to OpenRouter mid-life. The two
    sets of numbers are not comparable -- different provider, route and models --
    so the transport is part of the grouping key and they can only ever appear as
    separate rows."""
    rows = [_call(latency=10.0, transport="vercel-ai-gateway"),
            _call(latency=1000.0, transport="openrouter")]
    summary = report.summarise(rows)
    assert set(summary) == {"gpt/highway@vercel-ai-gateway", "gpt/highway@openrouter"}
    assert summary["gpt/highway@vercel-ai-gateway"]["p50_ms"] == 10.0
    assert summary["gpt/highway@openrouter"]["p50_ms"] == 1000.0


def test_the_transport_is_carried_onto_every_row_it_summarises():
    assert report.summarise([_call()])[KEY]["transport"] == "openrouter"


# --- the three metrics added with the migration ------------------------------


def test_the_tail_ratio_is_reported_next_to_the_percentiles():
    """689 p50 against 7,093 p95 is a 10x tail that a p50-only table hides."""
    rows = [_call(latency=float(v)) for v in list(range(100, 200)) + [7000]]
    summary = report.summarise(rows)[KEY]
    assert summary["tail_ratio"] == pytest.approx(summary["p95_ms"] / summary["p50_ms"])
    assert summary["tail_ratio"] > 1


def test_cost_per_correct_decision_is_the_headline_number():
    """Cost per call is misleading when accuracy differs. Half the price at half
    the accuracy is the same price per decision you can act on."""
    rows = [_call(correct=1, cost=1e-05) for _ in range(5)] + \
           [_call(correct=0, cost=1e-05) for _ in range(5)]
    summary = report.summarise(rows)[KEY]
    assert summary["accuracy"] == pytest.approx(0.5)
    assert summary["cost_per_correct_usd"] == pytest.approx(2e-05)


def test_cost_per_correct_decision_is_undefined_rather_than_infinite_at_zero_accuracy():
    rows = [_call(correct=0, cost=1e-05) for _ in range(4)]
    assert report.summarise(rows)[KEY]["cost_per_correct_usd"] is None


def test_cost_per_correct_decision_is_absent_where_the_task_has_no_gold():
    assert report.summarise([_call(correct=None)])[KEY]["cost_per_correct_usd"] is None


def test_the_repeat_pass_is_not_folded_into_the_measured_numbers():
    """`--repeat` re-asks a subset. Averaging those calls into the latency and cost
    of the run would silently weight the repeated examples twice."""
    rows = [_call(example_id="e1", latency=10.0, repeat=0),
            _call(example_id="e2", latency=20.0, repeat=0),
            _call(example_id="e1", latency=9999.0, repeat=1)]
    summary = report.summarise(rows)[KEY]
    assert summary["n"] == 2
    assert summary["p95_ms"] == 20.0


def test_determinism_is_the_share_of_repeated_examples_answered_differently():
    rows = [_call(example_id="e1", choice="IDLE", repeat=0),
            _call(example_id="e2", choice="IDLE", repeat=0),
            _call(example_id="e1", choice="IDLE", repeat=1),
            _call(example_id="e2", choice="SLOWER", repeat=1)]
    assert report.determinism(rows)[KEY] == {"disagreement_rate": 0.5, "n": 2}


def test_determinism_is_measured_on_the_repeated_subset_only():
    """The denominator is what was actually asked twice, and the report has to say
    so rather than implying the whole run was checked."""
    rows = [_call(example_id=f"e{i}", repeat=0) for i in range(10)] + \
           [_call(example_id="e0", choice="SLOWER", repeat=1)]
    assert report.determinism(rows)[KEY] == {"disagreement_rate": 1.0, "n": 1}


def test_a_run_with_no_repeat_pass_reports_no_determinism_rather_than_perfect():
    assert report.determinism([_call()]) == {}


def test_determinism_keeps_the_two_transports_apart_as_well():
    rows = [_call(transport="openrouter", repeat=0), _call(transport="openrouter", repeat=1),
            _call(transport="vercel-ai-gateway", repeat=0)]
    out = report.determinism(rows)
    assert set(out) == {"gpt/highway@openrouter"}


# --- input tokens for one identical prompt ------------------------------------


def test_identical_prompt_tokens_contrasts_what_each_model_is_billed_for_one_prompt():
    """The same text, so the difference is each model's tokenizer and protocol
    overhead rather than a difference in the prompt. The earlier pilot had one
    model at 197 input tokens and another at 637 for the same words."""
    config = {"option_overhead": {
        "jev/highway": {"prompt_tokens_with_options": 197, "n_options": 5},
        "gpt/highway": {"prompt_tokens_with_options": 637, "n_options": 5},
        "laya/highway": {"prompt_tokens_with_options": None, "n_options": 5},
    }}
    rows = report.identical_prompt_tokens(config, "highway")
    assert rows[0] == ("jev/highway", 197)
    assert rows[-1] == ("gpt/highway", 637)
    assert all(tokens is not None for _, tokens in rows)  # laya reports none, so it is left out


def test_identical_prompt_tokens_is_empty_when_nothing_was_probed():
    assert report.identical_prompt_tokens({}, "highway") == []

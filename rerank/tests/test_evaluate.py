"""What the task does between the store and the ranking: decide what is left to
score, turn the records back into rankings, and read the two things the
full-scale run exists to check -- what the calls cost, and how many passages the
scores leave tied."""
from rerank.evaluate import market_cost, pending, rankings_from_records, ties
import pytest

CANDIDATES = {"q1": ["d1", "d2"], "q2": ["d3"]}


def _rec(method, qid, doc, score):
    return {"method": method, "query_id": qid, "doc_id": doc, "score": score}


def test_everything_is_pending_on_a_fresh_run():
    assert pending(CANDIDATES, "laya-score", set()) == [("q1", "d1"), ("q1", "d2"), ("q2", "d3")]


def test_what_is_already_scored_is_not_scored_again():
    done = {("laya-score", "q1", "d1")}
    assert pending(CANDIDATES, "laya-score", done) == [("q1", "d2"), ("q2", "d3")]


def test_another_methods_records_do_not_count_as_done():
    done = {("cross-encoder", "q1", "d1")}
    assert pending(CANDIDATES, "laya-score", done) == [("q1", "d1"), ("q1", "d2"), ("q2", "d3")]


def test_rankings_reorder_the_candidates_by_score():
    records = [_rec("laya-score", "q1", "d1", 0.2), _rec("laya-score", "q1", "d2", 0.9),
               _rec("laya-score", "q2", "d3", 0.5)]
    assert rankings_from_records(CANDIDATES, records, "laya-score") == {"q1": ["d2", "d1"], "q2": ["d3"]}


def test_a_half_scored_query_is_left_out_rather_than_ranked_on_a_hole():
    records = [_rec("laya-score", "q1", "d1", 0.2), _rec("laya-score", "q2", "d3", 0.5)]
    assert rankings_from_records(CANDIDATES, records, "laya-score") == {"q2": ["d3"]}


def test_other_methods_records_are_ignored():
    records = [_rec("cross-encoder", "q2", "d3", 9.0), _rec("laya-score", "q2", "d3", 0.5)]
    assert rankings_from_records(CANDIDATES, records, "laya-score") == {"q2": ["d3"]}


def test_a_failed_call_keeps_its_bm25_position_rather_than_dropping_the_query():
    records = [_rec("jev-score", "q1", "d1", 0.2), {**_rec("jev-score", "q1", "d2", None), "failed": True},
               _rec("jev-score", "q2", "d3", 0.5)]
    assert rankings_from_records(CANDIDATES, records, "jev-score") == {"q1": ["d1", "d2"], "q2": ["d3"]}


def test_a_pair_that_was_never_scored_still_drops_the_query():
    """A hole and a recorded failure are different things: the failure is a fact
    about the model, a hole is an unfinished run."""
    records = [_rec("jev-score", "q1", "d1", 0.2), _rec("jev-score", "q2", "d3", 0.5)]
    assert rankings_from_records(CANDIDATES, records, "jev-score") == {"q2": ["d3"]}


def _priced(cost):
    return {"response": {"providerMetadata": {"gateway": {"marketCost": cost}}}}


def test_market_cost_is_read_from_the_gateways_own_per_call_field():
    """The cost of the run is the gateway's number, not a rate times a token
    count: `marketCost` is a string of dollars on every successful call."""
    assert market_cost(_priced("0.000024738")) == 0.000024738


def test_a_local_model_has_no_market_cost():
    assert market_cost({"response": {"logit": -1.0}}) == 0.0


def test_a_failed_call_has_no_market_cost():
    assert market_cost({"response": {"error": "HTTP 429: Too Many Requests"}}) == 0.0


def test_a_market_cost_that_is_not_a_number_counts_as_nothing_rather_than_raising():
    assert market_cost(_priced("n/a")) == 0.0


def _scored(qid, doc, score):
    return {"query_id": qid, "doc_id": doc, "score": score}


def test_distinct_scores_leave_no_ties():
    records = [_scored("q1", "d1", 0.1), _scored("q1", "d2", 0.2), _scored("q1", "d3", 0.3)]
    assert ties(records) == {"tied": 0, "largest_group": 0}


def test_passages_sharing_a_score_within_a_query_are_tied():
    """A tie is what holds a passage in its BM25 slot, so the count that matters
    is how many passages sit in a tie, not how many distinct values there are."""
    records = [_scored("q1", "d1", 0.5), _scored("q1", "d2", 0.5), _scored("q1", "d3", 0.9)]
    assert ties(records) == {"tied": 2, "largest_group": 2}


def test_the_same_score_under_different_queries_is_not_a_tie():
    """Only one query's candidates are ever sorted against each other."""
    records = [_scored("q1", "d1", 0.5), _scored("q2", "d2", 0.5)]
    assert ties(records) == {"tied": 0, "largest_group": 0}


def test_the_largest_group_is_the_worst_tie_not_the_total():
    records = [_scored("q1", "d1", 0.5), _scored("q1", "d2", 0.5), _scored("q1", "d3", 0.5),
               _scored("q1", "d4", 0.9), _scored("q1", "d5", 0.9)]
    assert ties(records) == {"tied": 5, "largest_group": 3}


def test_a_failed_call_is_not_tied_with_anything():
    """`None` is an absent opinion, not a score every failure shares."""
    records = [_scored("q1", "d1", None), _scored("q1", "d2", None), _scored("q1", "d3", 0.9)]
    assert ties(records) == {"tied": 0, "largest_group": 0}


class TestOutputLandsInsideTheExperiment:
    """Every experiment keeps its own evidence and results. rerank's used to sit
    at the repository root because it was the only experiment here; a default
    that writes outside `rerank/` would put them back."""

    def test_the_default_output_directory_is_the_experiment_not_the_repo_root(self):
        from rerank import evaluate
        assert evaluate.EXPERIMENT_DIR.name == "rerank"
        assert (evaluate.EXPERIMENT_DIR / "evaluate.py").exists()

    def test_the_export_script_reads_this_experiment_and_writes_the_repo_site(self):
        from rerank.scripts import export_examples
        assert export_examples.EXPERIMENT.name == "rerank"
        assert (export_examples.EXPERIMENT / "evaluate.py").exists()
        assert (export_examples.REPO / "site").is_dir(), "the site is the repo's, not the experiment's"
        assert export_examples.REPO.name == "ai-experiments"


class TestCostComesFromWhicheverProviderAnswered:
    """The price of a call is put on the call, and the two transports put it in
    different places. Reading only one of them reports a real run as free."""

    def test_openrouter_reports_it_under_usage(self):
        from rerank.evaluate import market_cost
        record = {"response": {"usage": {"input_tokens": 589, "output_tokens": 19,
                                         "cost": 2.4738e-05}}}
        assert market_cost(record) == pytest.approx(2.4738e-05)

    def test_the_gateway_reported_it_under_provider_metadata(self):
        """The recorded run used this shape; reading it keeps old logs scoreable."""
        from rerank.evaluate import market_cost
        record = {"response": {"providerMetadata": {"gateway": {"marketCost": "0.000025"}}}}
        assert market_cost(record) == pytest.approx(0.000025)

    def test_a_local_model_and_a_failed_call_cost_nothing(self):
        from rerank.evaluate import market_cost
        assert market_cost({"response": {"usage": {"input_tokens": 233}}}) == 0.0
        assert market_cost({"response": {"error": "HTTP 503"}}) == 0.0


class TestCallAccountingIsAlwaysReported:
    """A run with no retries and no failures still made calls and still cost
    money. Reporting the accounting only when something went wrong means a clean
    run prices itself at nothing."""

    def _summary(self, records):
        from rerank.evaluate import summarize
        qrels = {"q1": {"d1": 1}}
        return summarize("jev-score", records, qrels, {"q1": ["d1"]})

    def test_a_clean_hosted_run_still_reports_its_cost(self):
        records = [{"method": "jev-score", "query_id": "q1", "doc_id": "d1", "score": 2.0,
                    "latency_ms": 260.0,
                    "response": {"usage": {"input_tokens": 589, "cost": 2.4738e-05}}}]
        calls = self._summary(records)["calls"]
        assert calls["n"] == 1 and calls["retries"] == 0 and calls["failed"] == 0
        assert calls["market_cost_usd"] == pytest.approx(2.5e-05, abs=1e-6)
        assert calls["input_tokens"] == 589

    def test_a_local_model_reports_its_calls_and_a_true_zero(self):
        records = [{"method": "jev-score", "query_id": "q1", "doc_id": "d1", "score": 2.0,
                    "latency_ms": 900.0, "response": {"usage": {"input_tokens": 233}}}]
        calls = self._summary(records)["calls"]
        assert calls["n"] == 1 and calls["market_cost_usd"] == 0.0


class TestTheFloor:
    """The first stage's own order is what doing nothing gives you, so it is
    graded rather than scored: no model is asked anything for it."""

    def test_the_retrievers_order_is_taken_unchanged(self):
        from rerank.evaluate import rankings_for
        candidates = {"q1": ["d3", "d1", "d2"]}
        assert rankings_for("hybrid", candidates, []) == {"q1": ["d3", "d1", "d2"]}

    def test_it_needs_no_records(self):
        """Which is why it costs nothing and is always available."""
        from rerank.evaluate import FREE, rankings_for
        assert FREE == ("hybrid",)
        assert rankings_for("hybrid", {"q1": ["d1"]}, []) == {"q1": ["d1"]}



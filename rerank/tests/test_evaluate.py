"""What the task does between the store and the ranking: decide what is left to
score, turn the records back into rankings, and read the two things the
full-scale run exists to check -- what the calls cost, and how many passages the
scores leave tied."""
from rerank.evaluate import market_cost, pending, rankings_from_records, ties

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
        assert (export_examples.EXPERIMENT / "results").is_dir()
        assert (export_examples.REPO / "site").is_dir(), "the site is the repo's, not the experiment's"
        assert export_examples.REPO.name == "ai-experiments"

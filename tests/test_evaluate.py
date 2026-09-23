"""What the task does between the store and the ranking: decide what is left to
score, and turn the records back into rankings."""
from rerank.evaluate import pending, rankings_from_records

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

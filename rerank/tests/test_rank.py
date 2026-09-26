"""The scoring-to-ranking transformation, and the run format the metrics need."""
import pytest

from rerank.rank import build_run, rank_by_score, to_run


def test_orders_by_score_descending():
    assert rank_by_score(["a", "b", "c"], [0.1, 0.9, 0.5]) == ["b", "c", "a"]


def test_ties_keep_the_candidate_order():
    """Laya rounds scores to 4 decimals, so ties are common. A tie must not
    shuffle the passages: it falls back to the order BM25 handed us."""
    assert rank_by_score(["a", "b", "c"], [0.5, 0.5, 0.5]) == ["a", "b", "c"]
    assert rank_by_score(["a", "b", "c"], [0.5, 0.9, 0.5]) == ["b", "a", "c"]


def test_empty_candidates():
    assert rank_by_score([], []) == []


def test_mismatched_lengths_is_an_error():
    with pytest.raises(ValueError):
        rank_by_score(["a", "b"], [1.0])


def test_run_scores_are_strictly_descending_in_rank_order():
    run = to_run(["a", "b", "c"])
    assert run["a"] > run["b"] > run["c"]


def test_build_run_keys_every_query():
    run = build_run({"q1": ["a", "b"], "q2": ["c"]})
    assert set(run) == {"q1", "q2"}
    assert list(run["q1"]) == ["a", "b"]


def test_a_failed_call_holds_its_bm25_position():
    """A passage the model never scored keeps the slot BM25 gave it, and the
    passages that were scored are sorted into the slots that are left."""
    assert rank_by_score(["a", "b", "c"], [0.1, None, 0.9]) == ["c", "b", "a"]


def test_every_call_failing_leaves_the_candidate_order_untouched():
    assert rank_by_score(["a", "b", "c"], [None, None, None]) == ["a", "b", "c"]

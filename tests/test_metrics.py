"""The metric plumbing. pytrec_eval computes the numbers; these tests check we
hand it the right thing and read the right field back."""
import math

from rerank.metrics import evaluate, latency, mean_ci

QRELS = {"q1": {"a": 2, "b": 1}, "q2": {"z": 1}}


def test_perfect_ranking_scores_one():
    run = {"q1": {"a": 3.0, "b": 2.0, "c": 1.0}, "q2": {"z": 2.0, "y": 1.0}}
    m = evaluate(QRELS, run)
    assert m["ndcg@10"] == 1.0
    assert m["recall@10"] == 1.0
    assert m["mrr@10"] == 1.0


def test_a_relevant_passage_further_down_scores_lower():
    good = evaluate(QRELS, {"q1": {"a": 3.0, "b": 2.0}, "q2": {"z": 2.0, "y": 1.0}})
    worse = evaluate(QRELS, {"q1": {"b": 3.0, "a": 2.0}, "q2": {"y": 2.0, "z": 1.0}})
    assert worse["ndcg@10"] < good["ndcg@10"]
    assert worse["mrr@10"] < good["mrr@10"]


def test_mrr_is_cut_at_ten():
    """pytrec_eval's recip_rank is over the whole run; ours must be @10, so a
    relevant passage at rank 11 counts for nothing."""
    ranked = {f"d{i}": float(30 - i) for i in range(12)}
    ranked["z"] = 0.5  # rank 13
    m = evaluate({"q2": {"z": 1}}, {"q2": ranked})
    assert m["mrr@10"] == 0.0


def test_only_queries_in_both_are_scored():
    m = evaluate(QRELS, {"q1": {"a": 1.0}})
    assert m["queries"] == 1


def test_latency_percentiles():
    s = latency([10.0, 20.0, 30.0])
    assert s["p50_ms"] == 20.0
    assert s["n"] == 3
    assert s["p95_ms"] >= s["p50_ms"]


def test_latency_of_nothing():
    assert latency([]) == {"n": 0, "p50_ms": None, "p95_ms": None}


def test_mean_ci_brackets_the_mean():
    mean, low, high = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert math.isclose(mean, 2.5)
    assert low < mean < high


def test_mean_ci_of_one_value_has_no_interval():
    assert mean_ci([2.0]) == (2.0, None, None)

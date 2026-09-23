"""Retrieval metrics via pytrec_eval (the trec_eval C code), never by hand.

nDCG@10 is the headline; Recall@10 and MRR@10 are alongside it. trec_eval has no
MRR@10 measure -- `recip_rank` runs over the whole ranking -- so the run is cut
to the top k before it is asked for one.

`mean_ci` is copied from arena/arena/bench.py rather than imported: each demo in
this repo stands alone.
"""
import math
import statistics

import pytrec_eval

Z = 1.96  # 95%
# Two-sided 95% t critical values for small samples (df = n - 1); 1.96 beyond 30.
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(values):
    """Mean with a 95% t-interval. One value has no interval."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    t = T95[len(values) - 2] if len(values) - 1 <= len(T95) else Z
    half = t * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def _cut(run, k):
    return {qid: dict(sorted(docs.items(), key=lambda kv: -kv[1])[:k]) for qid, docs in run.items()}


def per_query(qrels, run, k=10):
    """{query_id: {"ndcg@10":..., "recall@10":..., "mrr@10":...}} for queries in both."""
    qrels = {qid: rels for qid, rels in qrels.items() if qid in run}
    if not qrels:
        return {}
    graded = pytrec_eval.RelevanceEvaluator(qrels, {f"ndcg_cut.{k}", f"recall.{k}"}).evaluate(run)
    reciprocal = pytrec_eval.RelevanceEvaluator(qrels, {"recip_rank"}).evaluate(_cut(run, k))
    return {qid: {f"ndcg@{k}": graded[qid][f"ndcg_cut_{k}"],
                  f"recall@{k}": graded[qid][f"recall_{k}"],
                  f"mrr@{k}": reciprocal[qid]["recip_rank"]} for qid in graded}


def evaluate(qrels, run, k=10):
    """Means over the queries that appear in both the qrels and the run."""
    scores = per_query(qrels, run, k)
    if not scores:
        return {f"ndcg@{k}": 0.0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, "queries": 0}
    keys = [f"ndcg@{k}", f"recall@{k}", f"mrr@{k}"]
    out = {key: round(statistics.fmean(s[key] for s in scores.values()), 4) for key in keys}
    return {**out, "queries": len(scores)}


def latency(values_ms):
    """p50 and p95 of one scoring call."""
    if not values_ms:
        return {"n": 0, "p50_ms": None, "p95_ms": None}
    p95 = statistics.quantiles(values_ms, n=20)[18] if len(values_ms) >= 2 else values_ms[0]
    return {"n": len(values_ms), "p50_ms": round(statistics.median(values_ms), 1), "p95_ms": round(p95, 1)}

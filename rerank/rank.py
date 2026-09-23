"""Turning scores into a ranking, and a ranking into what pytrec_eval reads.

The only interesting decision here is the tie-break. Laya rounds its answers to
four decimals, so identical scores happen often; when they do the passages keep
the order BM25 gave them, which makes "no opinion" mean "no change" instead of
"shuffle".
"""


def rank_by_score(doc_ids, scores):
    """Doc ids best-first. `doc_ids` is the candidate order; ties keep it."""
    if len(doc_ids) != len(scores):
        raise ValueError(f"{len(doc_ids)} candidates but {len(scores)} scores")
    return [doc for _, doc in sorted(enumerate(doc_ids), key=lambda p: (-scores[p[0]], p[0]))]


def to_run(ranking):
    """pytrec_eval reads scores, not positions, so the rank is written back as a
    strictly descending score. Only the order survives, which is the point."""
    return {doc: float(len(ranking) - i) for i, doc in enumerate(ranking)}


def build_run(rankings):
    """{query_id: [doc_id, ...]} -> the run dict pytrec_eval evaluates."""
    return {qid: to_run(ranking) for qid, ranking in rankings.items()}

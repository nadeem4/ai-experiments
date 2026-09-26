"""The CSV tables the report writes beside `summary.json`.

Everything this experiment measures is rows -- a query, a call, a distinct score
-- so everything but the run's provenance is a CSV. A 25,840-row table as JSON is
unreadable in a git diff, needs a parser to open, and is several times the size.
As CSV it diffs a row at a time and loads in one line anywhere.

`summary.json` keeps what is not rows: the dataset, the commit, the device, the
per-method aggregates and their intervals.
"""
import csv


def _write(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_scores(records, out_dir):
    """A row per (method, distinct score) with how many calls landed on it.

    This is the table behind "a tie is not an opinion": a scorer that answers on
    a coarse grid leaves passages tied, and tied passages keep the first stage's
    order, so
    part of the ranking credited to the model is still BM25's. The shape of this
    table is what separates a model answering in 401 steps from one answering in
    thousands."""
    counts = {}
    for record in records:
        if record["score"] is None:
            continue  # a call that never returned has no score to bin
        counts[(record["method"], record["score"])] = \
            counts.get((record["method"], record["score"]), 0) + 1
    rows = [{"method": method, "score": score, "calls": calls}
            for (method, score), calls in sorted(counts.items())]
    return _write(out_dir / "scores.csv", ["method", "score", "calls"], rows)


def write_latency(records, out_dir):
    """A row per successful call.

    p50 and p95 survive into the summary, but two numbers cannot be turned back
    into a distribution, and the tail is the thing that hurts in production. A
    failed call has no latency worth reporting: it is an outage, not a speed."""
    rows = [{"method": r["method"], "query_id": r["query_id"], "doc_id": r["doc_id"],
             "latency_ms": r["latency_ms"]}
            for r in records if not r.get("failed") and r["score"] is not None]
    return _write(out_dir / "latency.csv",
                  ["method", "query_id", "doc_id", "latency_ms"], rows)


def write_per_query(floor, deltas, relevant, out_dir):
    """A row per query: the first stage's own nDCG@10, each method's difference
    from it, and whether the query could have moved at all.

    `can_move` is the column that keeps the means honest. For many queries the
    first stage retrieves nothing relevant, so every method scores zero on them
    whatever order it picks; they contribute nothing but denominator."""
    methods = sorted(deltas)
    rows = []
    for query_id in sorted(floor):
        row = {"query_id": query_id, "floor_ndcg@10": floor[query_id],
               "relevant": relevant.get(query_id, 0),
               "can_move": bool(relevant.get(query_id, 0))}
        for method in methods:
            row[method] = deltas[method].get(query_id)
        rows.append(row)
    return _write(out_dir / "per-query.csv",
                  ["query_id", "floor_ndcg@10", "relevant", "can_move", *methods], rows)


def write_methods(summaries, out_dir):
    """A row per method: the headline table, flattened out of `summary.json`.

    The same numbers, in the shape you would put in a spreadsheet rather than the
    nested one a program wants."""
    rows = []
    for s in summaries:
        delta = s.get("ndcg@10_vs_floor") or {}
        interval = delta.get("ci95") or [None, None]
        rows.append({
            "method": s["method"],
            "ndcg@10": s.get("ndcg@10"), "recall@10": s.get("recall@10"),
            "mrr@10": s.get("mrr@10"),
            "vs_floor_mean": delta.get("mean"),
            "vs_floor_low": interval[0], "vs_floor_high": interval[1],
            "better": delta.get("better"), "worse": delta.get("worse"),
            "same": delta.get("same"),
            "latency_p50_ms": (s.get("latency") or {}).get("p50_ms"),
            "latency_p95_ms": (s.get("latency") or {}).get("p95_ms"),
            "calls": (s.get("calls") or {}).get("n"),
            "failed": (s.get("calls") or {}).get("failed"),
            "cost_usd": (s.get("calls") or {}).get("market_cost_usd"),
            "distinct_scores": (s.get("scores") or {}).get("distinct"),
            "tied": (s.get("scores") or {}).get("tied"),
        })
    return _write(out_dir / "methods.csv", list(rows[0]) if rows else ["method"], rows)

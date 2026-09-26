"""What the lab site reads, derived from the run.

    uv run python -m rerank.scripts.site_data

The page lets a reader pick any query and watch its twenty candidates re-order
under each method. That needs every query, which is only affordable because the
wire log does not ship: 25,840 records of roughly 2KB cannot go to a browser,
but the order each method produced can, and the order is the thing being shown.

So this writes the derived ranking, not the evidence. The evidence stays in
`runs/<tag>/scores.jsonl`, which is where a number is checked.
"""
import json
from pathlib import Path

EXPERIMENT = Path(__file__).resolve().parents[1]
REPO = EXPERIMENT.parent
TAG = "gpu"
FLOOR = "hybrid"
SNIPPET = 320  # enough to judge a passage by, far less than the 1,000 scored


def rank(doc_ids, scores):
    """Best first, ties keeping the candidate order -- the same rule the run used."""
    return sorted(doc_ids, key=lambda d: (-scores[d], doc_ids.index(d)))


def build(candidates, records, corpus, qrels, queries, per_query):
    """-> one payload, used by the tests and by `split` below."""
    by_method = {}
    for r in records:
        if r["score"] is None:
            continue
        by_method.setdefault(r["method"], {}).setdefault(r["query_id"], {})[r["doc_id"]] = r["score"]

    rows, wanted = [], set()
    for query_id, docs in candidates.items():
        wanted.update(docs)
        relevant = [d for d in docs if (qrels.get(query_id) or {}).get(d, 0) > 0]
        orders, scores = {FLOOR: list(docs)}, {}
        for method, per_q in by_method.items():
            got = per_q.get(query_id) or {}
            # A query only half scored cannot be ranked: ordering it on a hole
            # would show an order no method produced.
            if all(d in got for d in docs):
                orders[method] = rank(docs, got)
                scores[method] = got
        rows.append({
            "id": query_id, "text": queries[query_id], "orders": orders, "scores": scores,
            "relevant": relevant, "can_move": bool(relevant),
            "ndcg": (per_query.get(query_id) or {}),
        })

    docs = {d: {"title": corpus[d].get("title", ""),
                "text": corpus[d].get("text", "")[:SNIPPET]}
            for d in sorted(wanted) if d in corpus}
    return {"floor": FLOOR, "queries": sorted(rows, key=lambda r: r["id"]), "docs": docs}


def split(payload):
    """-> (index, detail).

    The index is imported into the page and must be small enough to paint with.
    The detail is fetched when the reader opens a query, because scores and
    passage text are most of the weight and none of it is needed until then.

    Scores travel as an array aligned to the order beside them rather than a map
    keyed by document id: the ids are already in the order, and repeating them
    was a third of the payload."""
    index_queries, detail_queries = [], {}
    for row in payload["queries"]:
        index_queries.append({k: row[k] for k in ("id", "text", "relevant", "can_move", "ndcg")})
        detail_queries[row["id"]] = {
            "orders": row["orders"],
            "scores": {m: [round(row["scores"][m][d], 4) for d in row["orders"][m]]
                       for m in row["scores"]},
        }
    index = {"floor": payload["floor"], "queries": index_queries,
             "titles": {d: v["title"] for d, v in payload["docs"].items()}}
    detail = {"queries": detail_queries,
              "passages": {d: v["text"] for d, v in payload["docs"].items()}}
    return index, detail


def write(candidates, records, corpus, qrels, queries, results_dir, run_dir):
    """-> [(path, bytes)]. Called by the report, which already has all of this."""
    import csv

    per_query = {}
    with open(results_dir / "per-query.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            per_query[row["query_id"]] = {
                k: float(v) for k, v in row.items()
                if k not in ("query_id", "relevant", "can_move") and v not in ("", None)}

    payload = build(candidates, records, corpus, qrels, queries, per_query)
    index, detail = split(payload)
    index["summary"] = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    index["config"] = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    written = []
    for path, data in ((REPO / "site" / "data" / "rerank.json", index),
                       (REPO / "site" / "public" / "rerank-detail.json", detail)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        written.append((path.name, path.stat().st_size))
    return written


def main(argv=None):
    from ..data import load_nfcorpus
    from ..store import load as load_records

    run = EXPERIMENT / "runs" / TAG
    corpus, queries, qrels = load_nfcorpus(None, "test")
    candidates = json.loads((run / "candidates.json").read_text(encoding="utf-8"))
    records = load_records(run / "scores.jsonl")
    for name, size in write(candidates, records, corpus, qrels, queries,
                            EXPERIMENT / "results" / TAG, run):
        print(f"  {name:<22} {size / 1024:>7.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

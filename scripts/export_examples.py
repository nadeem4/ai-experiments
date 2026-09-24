"""Export the site's data: the results table, and a curated slice of the wire.

    uv run python scripts/export_examples.py

The table on the site covers all 323 test queries, because it is read straight
out of `results/`. The browsable examples cannot: the run's wire log is 25,840
records of roughly 3 KB, far too much to ship to a browser. So this writes a
subset, chosen by `rerank.examples.select_examples` -- which is where the only
real decision lives, and which is tested -- and the site says on the page that
the examples are a subset and the numbers are not.

Nothing is summarised or rewritten on the way out. Each exported query carries
BM25's candidate order, every method's re-ordering of it, the official judgements,
and the exact request and response for every passage, byte for byte as recorded.
"""
import argparse
import json
import shutil
from pathlib import Path

from rerank.data import load_nfcorpus
from rerank.examples import select_examples
from rerank.metrics import per_query
from rerank.rank import build_run, rank_by_score
from rerank.store import load as load_records

ROOT = Path(__file__).resolve().parents[1]
FULL_RESULTS = "test-top20-q323-20260923-191610.json"
PILOT_RESULTS = "test-top50-q30-20260923-140116.json"
RUN_TAG = "test-top20-q323"
METHODS = ["jev-score", "cross-encoder", "laya-score", "laya-typed-score"]
SNIPPET_CHARS = 400  # what the list shows; the full passage is in the wire record


def rankings(candidates, by_pair, method):
    """Each method's re-ordering of the same candidate list. A pair with no
    record, or a failed call, is a `None` score: `rank_by_score` leaves that
    passage exactly where BM25 put it."""
    out = {}
    for qid, docs in candidates.items():
        scores = [(by_pair.get((method, qid, doc)) or {}).get("score") for doc in docs]
        out[qid] = rank_by_score(docs, scores)
    return out


def deltas(candidates, by_pair, qrels):
    """Per-query nDCG@10 for BM25 and for each method, as a difference."""
    floor = {qid: s["ndcg@10"] for qid, s in per_query(qrels, build_run(candidates)).items()}
    out = {}
    for method in METHODS:
        scored = per_query(qrels, build_run(rankings(candidates, by_pair, method)))
        out[method] = {qid: round(s["ndcg@10"] - floor[qid], 6) for qid, s in scored.items() if qid in floor}
    return out


def query_rows(candidates, by_pair, qrels):
    """One row per query for `select_examples`: how far Jev moved it, how far
    Laya moved it, and how many Jev calls never came back."""
    d = deltas(candidates, by_pair, qrels)
    rows = []
    for qid in sorted(candidates):
        failed = sum(1 for doc in candidates[qid]
                     if (by_pair.get(("jev-score", qid, doc)) or {}).get("failed"))
        rows.append({"query_id": qid, "jev_delta": d["jev-score"].get(qid, 0.0),
                     "laya_delta": d["laya-score"].get(qid, 0.0), "jev_failed": failed})
    return rows, d


def export_query(qid, reason, candidates, by_pair, corpus, queries, qrels, d):
    docs = candidates[qid]
    judged = {doc: grade for doc, grade in qrels.get(qid, {}).items() if doc in docs}
    methods, wire = {}, {}
    for method in METHODS:
        records = {doc: by_pair.get((method, qid, doc)) for doc in docs}
        scores = {doc: (r or {}).get("score") for doc, r in records.items()}
        methods[method] = {
            "order": rank_by_score(docs, [scores[doc] for doc in docs]),
            "scores": scores,
            "delta": d[method].get(qid, 0.0),
            "failed": [doc for doc, r in records.items() if r and r.get("failed")],
        }
        wire[method] = {doc: {k: r[k] for k in ("request", "response", "latency_ms", "retries", "failed") if k in r}
                        for doc, r in records.items() if r}
    return {
        "query_id": qid, "query": queries[qid], "reason": reason,
        "bm25": docs, "relevant": judged,
        "passages": {doc: {"title": corpus[doc].get("title", ""),
                           "snippet": corpus[doc].get("text", "")[:SNIPPET_CHARS],
                           "chars": len(corpus[doc].get("text", ""))} for doc in docs},
        "methods": methods, "wire": wire,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "site"), help="the site directory to write into")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    site = Path(args.out)
    data, public = site / "data", site / "public" / "examples"
    data.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(ROOT / "results" / FULL_RESULTS, data / "results.json")
    shutil.copyfile(ROOT / "results" / PILOT_RESULTS, data / "pilot.json")
    print(f"results {FULL_RESULTS} and pilot {PILOT_RESULTS} -> {data}")

    corpus, queries, qrels = load_nfcorpus(args.cache_dir)
    candidates = json.loads((ROOT / "runs" / RUN_TAG / "candidates.json").read_text())
    records = load_records(ROOT / "runs" / RUN_TAG / "scores.jsonl")
    by_pair = {(r["method"], r["query_id"], r["doc_id"]): r for r in records}
    print(f"{len(records)} wire records over {len(candidates)} queries")

    rows, d = query_rows(candidates, by_pair, qrels)
    picked = select_examples(rows)
    by_id = {r["query_id"]: r for r in rows}

    index = []
    for pick in picked:
        qid, row = pick["query_id"], by_id[pick["query_id"]]
        one = export_query(qid, pick["reason"], candidates, by_pair, corpus, queries, qrels, d)
        (public / f"{qid}.json").write_text(json.dumps(one), encoding="utf-8")
        index.append({"query_id": qid, "query": queries[qid], "reason": pick["reason"],
                      "jev_delta": row["jev_delta"], "laya_delta": row["laya_delta"],
                      "jev_failed": row["jev_failed"], "relevant": len(one["relevant"]),
                      "candidates": len(one["bm25"])})
    (data / "examples.json").write_text(json.dumps({
        "run": RUN_TAG, "queries_in_run": len(candidates), "records_in_run": len(records),
        "exported": index,
    }, indent=2), encoding="utf-8")

    total = sum(f.stat().st_size for f in public.glob("*.json"))
    print(f"{len(index)} queries exported, {total / 1e6:.1f} MB in {public}")
    for reason in dict.fromkeys(p["reason"] for p in picked):
        print(f"  {reason}: {[p['query_id'] for p in picked if p['reason'] == reason]}")


if __name__ == "__main__":
    main()

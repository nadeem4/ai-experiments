"""Re-rank BEIR NFCorpus candidates and score the result.

    uv run python -m rerank.evaluate --limit 30 --top-k 50

BM25 picks the candidates once; every method re-ranks that same list, so the
comparison is fair and BM25 itself is the floor. Every (query, passage) scored is
appended to runs/<tag>/scores.jsonl with the exact request and response, and that
file is also the resume point: rerun the same command after an interruption and
only the unscored pairs are sent.

Nothing is trained here. This is the zero-shot number.
"""
import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from . import store
from .bm25 import BM25Index
from .data import load_nfcorpus
from .metrics import evaluate as score_run
from .metrics import latency, mean_ci, per_query
from .rank import build_run, rank_by_score
from .rerankers import RERANKER_NAMES, make_reranker

METHODS = ["bm25"] + RERANKER_NAMES


def pending(candidates, method, done):
    """The (query, passage) pairs this method has not scored yet."""
    return [(qid, doc) for qid, docs in candidates.items() for doc in docs if (method, qid, doc) not in done]


def rankings_from_records(candidates, records, method):
    """Records back into {query_id: [doc_id best first]}. A query that is only
    half scored is left out rather than ranked on a hole."""
    scores = {(r["query_id"], r["doc_id"]): r["score"] for r in records if r["method"] == method}
    rankings = {}
    for qid, docs in candidates.items():
        values = [scores.get((qid, doc)) for doc in docs]
        if any(v is None for v in values):
            continue
        rankings[qid] = rank_by_score(docs, values)
    return rankings


def build_candidates(corpus, queries, top_k, log=lambda *_: None):
    started = time.perf_counter()
    index = BM25Index(corpus)
    log(f"BM25 index over {len(corpus)} documents in {time.perf_counter() - started:.1f}s")
    started = time.perf_counter()
    candidates = {qid: index.top_k(text, top_k) for qid, text in queries.items()}
    seconds = time.perf_counter() - started
    log(f"BM25 top-{top_k} for {len(queries)} queries in {seconds:.1f}s")
    return candidates, seconds


def score_method(reranker, candidates, queries, corpus, path, log=lambda *_: None):
    """Scores whatever is still pending, appending each record as it lands."""
    todo = pending(candidates, reranker.name, store.scored_keys(path))
    log(f"{reranker.name}: {len(todo)} pairs to score")
    for i, (qid, doc_id) in enumerate(todo, start=1):
        out = reranker.score(queries[qid], corpus[doc_id])
        store.append(path, {"method": reranker.name, "query_id": qid, "doc_id": doc_id,
                            "score": out["score"], "latency_ms": round(out["latency_ms"], 2),
                            "request": out["request"], "response": out["response"]})
        if i % 100 == 0 or i == len(todo):
            log(f"{reranker.name}: {i}/{len(todo)}")


def summarize(method, records, qrels, rankings, floor=None):
    """Metrics, latency, and whether the scores actually vary. A model that
    returns the same number for every passage is the failure mode to catch.

    `floor` is BM25's per-query nDCG@10. Every method re-ranks the same candidates
    for the same queries, so "did re-ranking improve the ranking" is a paired
    question, and the per-query difference answers it far more sharply than two
    overlapping intervals do."""
    mine = [r for r in records if r["method"] == method]
    values = [r["score"] for r in mine]
    run = build_run(rankings)
    out = {"method": method, **score_run(qrels, run)}
    scores = per_query(qrels, run)
    ndcg = {qid: s["ndcg@10"] for qid, s in scores.items()}
    if ndcg:
        _, low, high = mean_ci(list(ndcg.values()))
        out["ndcg@10_ci95"] = [round(low, 4), round(high, 4)] if low is not None else None
    if floor and ndcg and method != "bm25":
        shared = [qid for qid in ndcg if qid in floor]
        deltas = [ndcg[qid] - floor[qid] for qid in shared]
        mean, low, high = mean_ci(deltas)
        out["ndcg@10_vs_bm25"] = {"mean": round(mean, 4),
                                  "ci95": [round(low, 4), round(high, 4)] if low is not None else None,
                                  "better": sum(d > 0 for d in deltas), "worse": sum(d < 0 for d in deltas),
                                  "same": sum(d == 0 for d in deltas)}
    out["latency"] = latency([r["latency_ms"] for r in mine])
    out["scoring_wall_clock_s"] = round(sum(r["latency_ms"] for r in mine) / 1000, 1)
    if values:
        out["scores"] = {"distinct": len(set(values)), "of": len(values),
                         "min": round(min(values), 4), "max": round(max(values), 4),
                         "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0}
    return out


def usable_device():
    """The GPU can be visible but unusable (an old NVIDIA driver, for one), and
    every latency in the results depends on which one ran, so it is measured."""
    try:
        import torch
        if not torch.cuda.is_available():
            return "cpu"
        torch.zeros(1, device="cuda")
        return "cuda"
    except Exception:
        return "cpu"


def _commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return None


def format_table(summaries):
    head = (f"{'method':<14} {'nDCG@10':>8} {'ci95':>17} {'Recall@10':>10} {'MRR@10':>8} "
            f"{'p50 ms':>8} {'p95 ms':>8} {'wall s':>8}  scores")
    rows = [head, "-" * len(head)]
    for s in summaries:
        lat, sc, ci = s["latency"], s.get("scores"), s.get("ndcg@10_ci95")
        rows.append(f"{s['method']:<14} {s['ndcg@10']:>8.4f} "
                    f"{(f'[{ci[0]:.4f}-{ci[1]:.4f}]' if ci else ''):>17} "
                    f"{s['recall@10']:>10.4f} {s['mrr@10']:>8.4f} "
                    f"{str(lat['p50_ms']):>8} {str(lat['p95_ms']):>8} {s['scoring_wall_clock_s']:>8} "
                    + (f" {sc['distinct']} distinct of {sc['of']}, {sc['min']}..{sc['max']}" if sc else " (candidate order)"))
    deltas = [s for s in summaries if s.get("ndcg@10_vs_bm25")]
    if deltas:
        rows += ["", "nDCG@10 against the BM25 floor, paired per query:"]
        for s in deltas:
            d = s["ndcg@10_vs_bm25"]
            ci = f" [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d["ci95"] else ""
            rows.append(f"  {s['method']:<14} {d['mean']:+.4f}{ci}   better on {d['better']}, "
                        f"worse on {d['worse']}, unchanged on {d['same']} queries")
    return "\n".join(rows)


def run(limit, top_k, methods, out_dir, cache_dir=None, laya_path=None, split="test", log=print):
    corpus, queries, qrels = load_nfcorpus(cache_dir, split)
    log(f"NFCorpus {split}: {len(corpus)} documents, {len(queries)} queries with judgements")
    if limit:
        queries = {qid: queries[qid] for qid in sorted(queries)[:limit]}
        log(f"limited to the first {len(queries)} queries by id")

    tag = f"{split}-top{top_k}-q{len(queries)}"
    run_dir = Path(out_dir) / "runs" / tag
    candidates_path = run_dir / "candidates.json"
    scores_path = run_dir / "scores.jsonl"

    if candidates_path.exists():
        candidates = json.loads(candidates_path.read_text())
        bm25_seconds = None
        log(f"reusing {candidates_path}")
    else:
        candidates, bm25_seconds = build_candidates(corpus, queries, top_k, log)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_path.write_text(json.dumps(candidates))

    # BM25's per-query nDCG@10 is the floor every method is measured against, so
    # it is computed whether or not bm25 was asked for.
    floor = {qid: s["ndcg@10"] for qid, s in per_query(qrels, build_run(candidates)).items()}

    summaries = []
    for method in methods:
        if method != "bm25":
            started = time.perf_counter()
            reranker = make_reranker(method, **({"path": laya_path} if laya_path and method.startswith("laya") else {}))
            log(f"{method}: loaded in {time.perf_counter() - started:.1f}s")
            score_method(reranker, candidates, queries, corpus, scores_path, log)
        records = store.load(scores_path)
        rankings = candidates if method == "bm25" else rankings_from_records(candidates, records, method)
        summary = summarize(method, records, qrels, rankings, floor)
        if method == "bm25" and bm25_seconds is not None:
            summary["scoring_wall_clock_s"] = round(bm25_seconds, 1)
        summaries.append(summary)
        log(format_table([summary]))

    results = {"dataset": f"BEIR NFCorpus ({split})", "queries": len(queries), "top_k": top_k,
               "device": usable_device(), "commit": _commit(),
               "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "methods": summaries}
    results_path = Path(out_dir) / "results" / f"{tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    results["path"] = results_path
    return results


def main():
    p = argparse.ArgumentParser(description="Re-rank BEIR NFCorpus with a decision model and score it.")
    p.add_argument("--limit", type=int, default=30, help="queries (0 = all 323 in the test split)")
    p.add_argument("--top-k", type=int, default=50, help="BM25 candidates per query")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--split", default="test", choices=["test", "dev", "train"])
    p.add_argument("--out", default=".")
    p.add_argument("--cache-dir", default=None, help="where the dataset is cached (default: the HF cache)")
    p.add_argument("--laya-path", default=None, help="Laya weights folder (default: ../arena/models/laya)")
    args = p.parse_args()

    results = run(args.limit, args.top_k, args.methods, args.out, args.cache_dir, args.laya_path, args.split)
    print()
    print(format_table(results["methods"]))
    print(f"Saved {results['path']}")


if __name__ == "__main__":
    main()

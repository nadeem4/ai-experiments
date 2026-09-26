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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import figures, store, tables
from .bm25 import BM25Index
from .data import load_nfcorpus
from .metrics import evaluate as score_run
from .metrics import latency, mean_ci, per_query
from .rank import build_run, rank_by_score
from .rerankers import RERANKER_NAMES, make_reranker
# Every experiment keeps its evidence and its results inside itself. rerank's
# used to sit at the repository root because it was the only experiment here.
EXPERIMENT_DIR = Path(__file__).resolve().parent

METHODS = ["bm25"] + RERANKER_NAMES


def pending(candidates, method, done):
    """The (query, passage) pairs this method has not scored yet."""
    return [(qid, doc) for qid, docs in candidates.items() for doc in docs if (method, qid, doc) not in done]


def rankings_from_records(candidates, records, method):
    """Records back into {query_id: [doc_id best first]}. A query that is only
    half scored is left out rather than ranked on a hole -- but a pair that was
    scored and failed is not a hole: it is recorded, and its passage keeps the
    BM25 position."""
    scores = {(r["query_id"], r["doc_id"]): r["score"] for r in records if r["method"] == method}
    rankings = {}
    for qid, docs in candidates.items():
        if any((qid, doc) not in scores for doc in docs):
            continue
        rankings[qid] = rank_by_score(docs, [scores[(qid, doc)] for doc in docs])
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
        record = {"method": reranker.name, "query_id": qid, "doc_id": doc_id,
                  "score": out["score"], "latency_ms": round(out["latency_ms"], 2),
                  "request": out["request"], "response": out["response"]}
        if out.get("retries"):
            record["retries"] = out["retries"]
        if out.get("failed"):
            record["failed"] = True
        store.append(path, record)
        if i % 100 == 0 or i == len(todo):
            log(f"{reranker.name}: {i}/{len(todo)}")


def _input_tokens(record):
    """Laya reports `input_tokens`, the AI Gateway reports `inputTokens`."""
    usage = record["response"].get("usage") or {}
    return usage.get("input_tokens") or usage.get("inputTokens") or 0


def market_cost(record):
    """What the provider itself charged for this one call, in dollars.

    Not a token count multiplied by a rate from a pricing page: the provider puts
    the price of the call on the call, and the run's cost is the sum of those. A
    local model has none, and neither does a call that failed.

    The two transports report it in different places -- OpenRouter under
    `usage.cost`, the Vercel AI Gateway under `providerMetadata.gateway.marketCost`
    -- so both are read. Reading only one of them prices a real run at zero, which
    is a number that looks like an answer."""
    response = record["response"]
    usage = response.get("usage") or {}
    gateway = (response.get("providerMetadata") or {}).get("gateway") or {}
    try:
        return float(usage.get("cost") or gateway.get("marketCost") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def ties(records):
    """Passages sharing a score with another candidate for the same query.

    A model that answers in coarse steps leaves passages tied, and a tie is not
    an opinion: `rank_by_score` leaves those passages in the order BM25 gave
    them. So the count that says what a distinct-value count means is how many
    passages sit in a tie, and how big the worst one is."""
    by_query = {}
    for r in records:
        if r["score"] is not None:
            by_query.setdefault(r["query_id"], []).append(r["score"])
    tied = largest = 0
    for scores in by_query.values():
        for size in Counter(scores).values():
            if size > 1:
                tied += size
                largest = max(largest, size)
    return {"tied": tied, "largest_group": largest}


def summarize(method, records, qrels, rankings, floor=None):
    """Metrics, latency, and whether the scores actually vary. A model that
    returns the same number for every passage is the failure mode to catch.

    `floor` is BM25's per-query nDCG@10. Every method re-ranks the same candidates
    for the same queries, so "did re-ranking improve the ranking" is a paired
    question, and the per-query difference answers it far more sharply than two
    overlapping intervals do."""
    mine = [r for r in records if r["method"] == method]
    values = [r["score"] for r in mine if r["score"] is not None]
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
    ok = [r for r in mine if not r.get("failed")]
    out["latency"] = latency([r["latency_ms"] for r in ok])
    out["scoring_wall_clock_s"] = round(sum(r["latency_ms"] for r in ok) / 1000, 1)
    # Always, not only when something went wrong. A clean run still made calls
    # and still cost money, and reporting the accounting only on failure prices
    # a successful run at nothing.
    if mine:
        out["calls"] = {"n": len(mine), "retries": sum(r.get("retries", 0) for r in mine),
                        "failed": sum(bool(r.get("failed")) for r in mine),
                        "input_tokens": sum(_input_tokens(r) for r in ok),
                        "market_cost_usd": round(sum(market_cost(r) for r in mine), 6)}
    if values:
        out["scores"] = {"distinct": len(set(values)), "of": len(values),
                         "min": round(min(values), 4), "max": round(max(values), 4),
                         "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
                         **ties(mine)}
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
    head = (f"{'method':<18} {'nDCG@10':>8} {'ci95':>17} {'Recall@10':>10} {'MRR@10':>8} "
            f"{'p50 ms':>8} {'p95 ms':>8} {'wall s':>8}  scores")
    rows = [head, "-" * len(head)]
    for s in summaries:
        lat, sc, ci = s["latency"], s.get("scores"), s.get("ndcg@10_ci95")
        rows.append(f"{s['method']:<18} {s['ndcg@10']:>8.4f} "
                    f"{(f'[{ci[0]:.4f}-{ci[1]:.4f}]' if ci else ''):>17} "
                    f"{s['recall@10']:>10.4f} {s['mrr@10']:>8.4f} "
                    f"{str(lat['p50_ms']):>8} {str(lat['p95_ms']):>8} {s['scoring_wall_clock_s']:>8} "
                    + (f" {sc['distinct']} distinct of {sc['of']}, {sc['min']}..{sc['max']}, {sc['tied']} tied"
                       if sc else " (candidate order)")
                    + (f"; {s['calls']['retries']} retries, {s['calls']['failed']} failed, "
                       f"${s['calls']['market_cost_usd']:.4f}" if s.get("calls") else ""))
    deltas = [s for s in summaries if s.get("ndcg@10_vs_bm25")]
    if deltas:
        rows += ["", "nDCG@10 against the BM25 floor, paired per query:"]
        for s in deltas:
            d = s["ndcg@10_vs_bm25"]
            ci = f" [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]" if d["ci95"] else ""
            rows.append(f"  {s['method']:<18} {d['mean']:+.4f}{ci}   better on {d['better']}, "
                        f"worse on {d['worse']}, unchanged on {d['same']} queries")
    return "\n".join(rows)


def _setup(limit, top_k, out_dir, tag, split, cache_dir, log):
    """Everything both phases need: the data, the candidate set, and where they live.

    The candidate set is built once and pinned to disk, so the reporting phase
    re-ranks exactly the passages that were scored rather than rebuilding a set
    that might differ."""
    corpus, queries, qrels = load_nfcorpus(cache_dir, split)
    log(f"NFCorpus {split}: {len(corpus)} documents, {len(queries)} queries with judgements")
    if limit:
        queries = {qid: queries[qid] for qid in sorted(queries)[:limit]}
        log(f"limited to the first {len(queries)} queries by id")

    run_dir = Path(out_dir) / "runs" / tag
    candidates_path, config_path = run_dir / "candidates.json", run_dir / "config.json"

    if candidates_path.exists():
        candidates = json.loads(candidates_path.read_text())
        log(f"reusing {candidates_path}")
    else:
        candidates, bm25_seconds = build_candidates(corpus, queries, top_k, log)
        candidates_path.parent.mkdir(parents=True, exist_ok=True)
        candidates_path.write_text(json.dumps(candidates))
        # BM25's own wall clock is measured while building the candidates and is
        # wanted by the report, which runs later and never builds them. Without
        # persisting it here the number is simply lost.
        config_path.write_text(json.dumps({
            "tag": tag, "split": split, "top_k": top_k, "limit": limit,
            "queries": len(queries),
            "bm25_seconds": round(bm25_seconds, 1),
        }, indent=2))

    return corpus, queries, qrels, run_dir, candidates


def score(limit, top_k, methods, out_dir, tag, split="test", cache_dir=None,
          laya_path=None, log=print):
    """Phase one: ask the models. Appends to the wire log and writes nothing else.

    A method whose pairs are all already in the store is skipped entirely -- the
    model is never constructed, so a fully-scored re-run needs no API key, no
    network and no weights on disk."""
    corpus, queries, qrels, run_dir, candidates = _setup(
        limit, top_k, out_dir, tag, split, cache_dir, log)
    scores_path = run_dir / "scores.jsonl"

    for method in methods:
        if method == "bm25":
            continue
        if not pending(candidates, method, store.scored_keys(scores_path)):
            log(f"{method}: every pair is already in the store, nothing to score")
            continue
        started = time.perf_counter()
        reranker = make_reranker(
            method, **({"path": laya_path} if laya_path and method.startswith("laya") else {}))
        log(f"{method}: loaded in {time.perf_counter() - started:.1f}s")
        score_method(reranker, candidates, queries, corpus, scores_path, log)


def report(limit, top_k, methods, out_dir, tag, split="test", cache_dir=None, log=print):
    """Phase two: read the wire log and write everything that is published.

    Loads no model and makes no network call, which is what lets it be re-run
    for free every time a number's definition changes."""
    corpus, queries, qrels, run_dir, candidates = _setup(
        limit, top_k, out_dir, tag, split, cache_dir, log)
    scores_path = run_dir / "scores.jsonl"
    config = json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {}

    # BM25's per-query nDCG@10 is the floor every method is measured against, so
    # it is computed whether or not bm25 was asked for.
    floor = {qid: s["ndcg@10"] for qid, s in per_query(qrels, build_run(candidates)).items()}
    records = store.load(scores_path)

    summaries, deltas = [], {}
    for method in methods:
        rankings = candidates if method == "bm25" else rankings_from_records(candidates, records, method)
        summary = summarize(method, records, qrels, rankings, floor)
        if method == "bm25" and config.get("bm25_seconds") is not None:
            summary["scoring_wall_clock_s"] = config["bm25_seconds"]
        summaries.append(summary)
        if method != "bm25":
            scored = per_query(qrels, build_run(rankings))
            deltas[method] = {qid: round(s["ndcg@10"] - floor[qid], 4)
                              for qid, s in scored.items() if qid in floor}
        log(format_table([summary]))

    results = {"dataset": f"BEIR NFCorpus ({split})", "tag": tag, "queries": len(queries),
               "top_k": top_k, "device": usable_device(), "commit": _commit(),
               "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "methods": summaries}
    results_dir = Path(out_dir) / "results" / tag
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")

    # How many of a query's candidates the qrels actually judge relevant. A query
    # with none cannot be re-ranked into a better score by anybody, so the tables
    # carry the count rather than leaving it to be rediscovered.
    relevant = {qid: sum(1 for doc in docs if qrels.get(qid, {}).get(doc, 0) > 0)
                for qid, docs in candidates.items()}
    written = [tables.write_methods(summaries, results_dir),
               tables.write_per_query(floor, deltas, relevant, results_dir),
               tables.write_scores(records, results_dir),
               tables.write_latency(records, results_dir)]
    for path in written:
        log(f"  table: {path}")

    figure_dir = results_dir / "figures"
    drawn = figures.write_all(results, deltas, relevant, records, figure_dir)
    for name in drawn["written"]:
        log(f"  figure: {figure_dir / name}")
    for name, reason in drawn["skipped"].items():
        log(f"  figure {name} SKIPPED: {reason}")

    results["path"] = results_dir / "summary.json"
    return results


def main(argv=None):
    """The direct entry point. `uv run exp run rerank` is the same two calls."""
    p = argparse.ArgumentParser(description="Re-rank BEIR NFCorpus with a decision model and score it.")
    p.add_argument("--tag", default="pilot", help="names runs/<tag>/ and results/<tag>/")
    p.add_argument("--limit", type=int, default=0, help="queries (0 = all 323 in the test split)")
    p.add_argument("--top-k", type=int, default=20, help="BM25 candidates per query")
    p.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    p.add_argument("--split", default="test", choices=["test", "dev", "train"])
    p.add_argument("--out", default=str(EXPERIMENT_DIR),
                   help="write runs/ and results/ under here (default: the experiment)")
    p.add_argument("--cache-dir", default=None, help="where the dataset is cached (default: the HF cache)")
    p.add_argument("--laya-path", default=None, help="Laya weights folder (default: $LAYA_PATH, else models/laya, downloaded on first use)")
    args = p.parse_args(argv)

    score(args.limit, args.top_k, args.methods, args.out, args.tag, args.split,
          args.cache_dir, args.laya_path)
    results = report(args.limit, args.top_k, args.methods, args.out, args.tag,
                     args.split, args.cache_dir)
    print()
    print(format_table(results["methods"]))
    print(f"Saved {results['path']}")


if __name__ == "__main__":
    main()

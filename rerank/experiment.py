"""How `exp` runs this experiment.

The two phases stay separate functions even though one command calls both:
`run` talks to the models and appends to the wire log, `report` reads that log
and writes everything else. Keeping them apart is what lets a second invocation
skip the scoring entirely and just recompute the tables.
"""
import json
from pathlib import Path

from . import evaluate

NAME = "rerank"
TITLE = "Can a decision model re-rank retrieval better than BM25?"
COSTS_MONEY = True  # the Jev pass is hosted; everything else runs locally
EXPERIMENT_DIR = Path(__file__).resolve().parent


def add_run_arguments(parser):
    parser.add_argument("--top-k", type=int, default=None,
                        help="BM25 candidates per query (default: 20, or what this tag used)")
    parser.add_argument("--methods", nargs="+", default=evaluate.METHODS,
                        choices=evaluate.METHODS)
    parser.add_argument("--split", default=None, choices=["test", "dev", "train"])
    parser.add_argument("--cache-dir", default=None,
                        help="where the dataset is cached (default: the HF cache)")
    parser.add_argument("--laya-path", default=None,
                        help="Laya weights folder (default: $LAYA_PATH, else downloaded)")


def out_dir(args):
    return Path(getattr(args, "out", None) or EXPERIMENT_DIR)


def recorded_invocation(args):
    """What this tag was run with last time, or {} if it has not run.

    A tag names a measurement, so re-running it must mean the same measurement:
    a different --top-k is a different candidate set and therefore a different
    experiment, not a resumption of this one."""
    path = out_dir(args) / "runs" / args.tag / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _settings(args):
    was = recorded_invocation(args)
    return {
        "limit": args.limit if args.limit is not None else was.get("limit", 0),
        "top_k": args.top_k if args.top_k is not None else was.get("top_k", 20),
        "split": args.split or was.get("split") or "test",
        "methods": args.methods,
        "tag": args.tag,
        "out_dir": out_dir(args),
    }


def tags():
    """Which tags already have results, for `exp list`."""
    results = EXPERIMENT_DIR / "results"
    return sorted(p.name for p in results.iterdir() if p.is_dir()) if results.is_dir() else []


def run(args):
    evaluate.score(**_settings(args), cache_dir=args.cache_dir, laya_path=args.laya_path)


def report(args):
    evaluate.report(**_settings(args), cache_dir=args.cache_dir)


def dataset():
    """What NFCorpus actually holds, read off the files the run loads."""
    from collections import Counter

    from .data import load_nfcorpus

    corpus, queries, qrels = load_nfcorpus(None, "test")
    judged = [len(v) for v in qrels.values()]
    grades = Counter(g for v in qrels.values() for g in v.values())
    # A query that reads as a question, and a document judged relevant to it.
    # Taking the first query alphabetically gives "deafness", which shows a
    # reader nothing about what the data is.
    first = next(q for q in sorted(queries)
                 if qrels.get(q) and len(queries[q].split()) >= 4)
    doc = sorted(qrels[first])[0]
    return [{
        "name": "BEIR NFCorpus",
        "source": "BeIR/nfcorpus",
        "url": "https://huggingface.co/datasets/BeIR/nfcorpus",
        "licence": None,
        "what_it_is": "Health questions from NutritionFacts.org over PubMed abstracts, "
                      "with human relevance judgements. A query is a question a person "
                      "asked; a document is a paper that may or may not answer it.",
        "splits": {"documents": len(corpus), "test queries": len(queries)},
        "used": "`test`, all 323 queries",
        "classes": [],
        "extra": {
            "Judgements": f"{sum(judged):,} query-document pairs marked relevant, "
                          f"a median of {sorted(judged)[len(judged) // 2]} per query",
            "Grades": f"1 or 2, never 0: {grades[1]:,} marked relevant and "
                      f"{grades[2]:,} highly so. Anything above 0 counts as relevant here",
        },
        "rows": [
            {"query_id": first, "query": queries[first]},
            {"a document judged relevant to it": doc,
             "title": corpus[doc]["title"], "text": corpus[doc]["text"]},
        ],
    }]

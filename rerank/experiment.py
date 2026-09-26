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

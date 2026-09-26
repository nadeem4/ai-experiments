"""How `exp` runs this experiment.

`run` and `report` are the existing two entry points; one command calls both, and
`run` skips any arm already complete in the wire log.
"""
import json
from pathlib import Path

from . import arms as arms_mod
from . import report as report_module
from . import run as run_module

NAME = "banking77"
TITLE = "Does Laya's Banking77 failure come from its token budget?"
STATUS = "complete"
COSTS_MONEY = False  # Laya runs on local CPU; nothing here is hosted
EXPERIMENT_DIR = Path(__file__).resolve().parent


def add_run_arguments(parser):
    parser.add_argument("--arms", default="all",
                        help="comma-separated arm names, or 'all'")
    parser.add_argument("--seed", type=int, default=arms_mod.SEED)


def out_dir(args):
    return Path(getattr(args, "out", None) or EXPERIMENT_DIR)


def tags():
    results = EXPERIMENT_DIR / "results"
    return sorted(p.name for p in results.iterdir() if p.is_dir()) if results.is_dir() else []


def recorded_invocation(args):
    """What this tag was run with last time, or {} if it has not run.

    A tag names a measurement, so re-running it must mean the same measurement
    rather than whatever the flag defaults happen to be today."""
    path = out_dir(args) / "runs" / args.tag / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def run(args):
    was = recorded_invocation(args)
    limit = args.limit if args.limit is not None else was.get("limit", 0)
    run_module.main(["--tag", args.tag, "--limit", str(limit),
                     "--arms", args.arms, "--seed", str(args.seed),
                     "--out", str(out_dir(args))])


def report(args):
    report_module.main(["--tag", args.tag, "--seed", str(args.seed),
                        "--out", str(out_dir(args))])

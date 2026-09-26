"""How `exp` runs this experiment.

`run` and `report` are the existing two entry points; one command calls both, and
`run` skips every call already in the wire log -- which is why a second
invocation needs no API key.
"""
import json
from pathlib import Path

from . import report as report_module
from . import run as run_module

NAME = "typed_decisions"
TITLE = "What does a typed decision cost at 4 options and at 151?"
STATUS = "complete"
COSTS_MONEY = True  # every hosted call goes over OpenRouter
EXPERIMENT_DIR = Path(__file__).resolve().parent


def add_run_arguments(parser):
    parser.add_argument("--models", default=None,
                        help="comma-separated, or name=model-id to pin a build")
    parser.add_argument("--tasks", default=None, help="comma-separated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--arms", default=None,
                        help="which arms to run: main, bias, validation")
    parser.add_argument("--bias-subset", type=int, default=None,
                        help="how many leading examples the position-bias arms cover")
    parser.add_argument("--validation", type=int, default=None,
                        help="rows carved out of TRAIN to fit the calibration temperature on")
    parser.add_argument("--yes-spend", action="store_true",
                        help="confirm a run estimated above the cost guard")


def out_dir(args):
    return Path(getattr(args, "out", None) or EXPERIMENT_DIR)


def tags():
    results = EXPERIMENT_DIR / "results"
    return sorted(p.name for p in results.iterdir() if p.is_dir()) if results.is_dir() else []


def recorded_invocation(args):
    """What this tag was run with last time, or {} if it has not run.

    A tag names a measurement, and re-running it must mean the same measurement.
    Without this, `exp run typed_decisions --tag full` would take the flag
    defaults rather than the 300-example limit the run actually used, decide that
    every call was missing, and offer to spend the money again."""
    path = out_dir(args) / "runs" / args.tag / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("last_invocation") or {}


def _optional(flag, value):
    return [flag, str(value)] if value is not None else []


def run(args):
    was = recorded_invocation(args)
    run_module.main([
        "--tag", args.tag,
        "--limit", str(args.limit if args.limit is not None else was.get("limit", 0)),
        "--models", args.models or was.get("models") or "all",
        "--tasks", args.tasks or was.get("tasks") or "ag_news,clinc150",
        "--seed", str(args.seed),
        "--arms", args.arms or was.get("arms") or "main,bias,validation",
        "--out", str(out_dir(args)),
        *_optional("--bias-subset", args.bias_subset),
        *_optional("--validation", args.validation),
        *(["--yes-spend"] if args.yes_spend else []),
    ])


def report(args):
    report_module.main(["--tag", args.tag, "--out", str(out_dir(args))])

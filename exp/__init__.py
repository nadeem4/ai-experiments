"""One command to run any experiment in this repository.

    uv run exp list
    uv run exp run rerank --tag full --top-k 20

Two verbs, because an experiment only has two things you want from it: the list
of what is here, and "produce this experiment's results".

**`run` is idempotent, and that is why there is no separate report command.**
It scores whatever is missing from `<experiment>/runs/<tag>/`, then always
computes the tables and figures from that log. Run it a second time and the
scoring is skipped, so fixing how a number is computed costs seconds rather than
re-running an experiment that takes hours and costs money. Re-deriving results
from an untouched wire log is the common case -- far more common than measuring
again -- and it needs no second verb, only a run that does not redo finished
work.

Each experiment owns its own flags. `exp` contributes the four that it needs to
do its job (`--tag`, `--out`, `--limit`, `--dry-run`) and the experiment adds the
rest through `add_run_arguments`, so `uv run exp run <name> --help` always shows
the real surface and there is no passthrough syntax to remember.
"""
import argparse
import importlib
import sys

# Explicit, because a directory listing is not a decision. An experiment appears
# here when it is ready to be run by name, and adding one changes nothing else.
EXPERIMENTS = ("rerank", "banking77", "typed_decisions")


def load(name):
    """-> the experiment's `experiment` module, or a SystemExit naming what exists."""
    if name not in EXPERIMENTS:
        raise SystemExit(f"no experiment called {name!r}. This repository holds: "
                         f"{', '.join(EXPERIMENTS)}")
    return importlib.import_module(f"{name}.experiment")


def add_standard_arguments(parser):
    """The flags `exp` itself needs, which every experiment therefore gets.

    Kept to four on purpose: each one exists because the CLI cannot work without
    it, not because it sounded universal."""
    parser.add_argument("--tag", default="pilot",
                        help="names runs/<tag>/ and results/<tag>/ (default: pilot)")
    parser.add_argument("--out", default=None,
                        help="write under this directory instead of the experiment's own")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N examples (0 = all). Left out, a tag that has "
                             "run before reuses the number it used then")
    parser.add_argument("--dry-run", action="store_true",
                        help="say what would be measured and what is already done, then stop")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="exp", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="every experiment, its status, and which tags have results")

    run = sub.add_parser("run", help="produce an experiment's results")
    run.add_argument("name", choices=EXPERIMENTS)
    add_standard_arguments(run)

    # The experiment's own flags are added once its name is known, so --help
    # after a name shows that experiment's real surface rather than a union of
    # all three. The name is read straight off argv rather than pre-parsed:
    # argparse would act on a --help in there and exit before the flags existed.
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        name = next((a for a in argv[1:] if not a.startswith("-")), None)
        if name in EXPERIMENTS:
            load(name).add_run_arguments(run)

    args = parser.parse_args(argv)
    if args.command == "list":
        return list_experiments()
    return run_experiment(args)


def list_experiments():
    for name in EXPERIMENTS:
        experiment = load(name)
        tags = ", ".join(experiment.tags()) or "(none yet)"
        print(f"{name:<18} {experiment.STATUS:<10} {experiment.TITLE}")
        print(f"{'':<18} {'':<10} results: {tags}")
    return 0


def run_experiment(args):
    experiment = load(args.name)
    experiment.run(args)
    experiment.report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

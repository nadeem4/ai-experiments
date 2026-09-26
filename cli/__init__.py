"""One command to run any experiment in this repository.

    uv run cli list
    uv run cli run    rerank --tag full --top-k 20     # here
    uv run cli submit rerank                           # on a Kaggle GPU
    uv run cli fetch  rerank                           # bring the results back

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

Each experiment owns its own flags. `cli` contributes the four that it needs to
do its job (`--tag`, `--out`, `--limit`, `--dry-run`) and the experiment adds the
rest through `add_run_arguments`, so `uv run cli run <name> --help` always shows
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
    parser = argparse.ArgumentParser(prog="cli", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="every experiment, its status, and which tags have results")

    run = sub.add_parser("run", help="produce an experiment's results, here")
    run.add_argument("name", choices=EXPERIMENTS)
    add_standard_arguments(run)

    submit = sub.add_parser("submit", help="run it on a Kaggle GPU instead")
    submit.add_argument("name", choices=EXPERIMENTS)
    submit.add_argument("--dry-run", action="store_true",
                        help="check the prerequisites and stop")
    state = sub.add_parser("status", help="what the Kaggle run is doing")
    state.add_argument("name", choices=EXPERIMENTS)
    fetch = sub.add_parser("fetch", help="download a finished Kaggle run")
    fetch.add_argument("name", choices=EXPERIMENTS)
    fetch.add_argument("-p", "--path", default=None)

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
    if args.command in ("submit", "status", "fetch"):
        from . import kaggle

        if args.command == "submit":
            return kaggle.submit(args.name, args.dry_run)
        if args.command == "status":
            return kaggle.status(args.name)
        return kaggle.fetch(args.name, args.path)
    return run_experiment(args)


def list_experiments():
    for name in EXPERIMENTS:
        try:
            experiment = load(name)
        except ModuleNotFoundError:
            # A checkout can hold one experiment and not the others, which is
            # what a Kaggle kernel does. Say so rather than fail the listing.
            print(f"{name:<18} {'not checked out':<10}")
            continue
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

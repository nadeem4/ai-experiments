"""How `cli` runs the emeans experiment.

Slice E1 only lays the scaffold: the contract `cli` needs, the two scales and
`--dry-run`. The parts themselves arrive in later slices.
"""
from pathlib import Path
from typing import NamedTuple, Optional

NAME = "emeans"
TITLE = "Does e-means keep a growing vector index fresh more cheaply than rebuilding it?"
COSTS_MONEY = False  # everything timed runs on the local CPU
EXPERIMENT_DIR = Path(__file__).resolve().parent


class Scale(NamedTuple):
    n: Optional[int]  # base vectors used; None = all of them
    n_test: int
    n_queries: int
    k: int
    seeds_a: int
    seeds_bc: int
    b_months: int
    nprobes: tuple  # search settings; the pilot adds k so nprobe = k must give recall 1.0


PILOT = Scale(n=100_000, n_test=10_000, n_queries=1_000, k=256, seeds_a=1, seeds_bc=1, b_months=3,
              nprobes=(1, 4, 16, 64, 256))
FULL = Scale(n=None, n_test=100_000, n_queries=10_000, k=4096, seeds_a=5, seeds_bc=3, b_months=72,
             nprobes=(1, 4, 16, 64))


def scale_for(tag):
    """`pilot` is the pilot scale and `full` the full one; anything else is
    refused, because a typo must not start a multi-hour run."""
    if tag == "pilot":
        return PILOT
    if tag == "full":
        return FULL
    raise SystemExit(f"--tag must be 'pilot' or 'full' for emeans, not {tag!r}")


def add_run_arguments(parser):
    parser.add_argument("--part", choices=["a", "b", "c"], default="a",
                        help="a: the race, b: the growing index, c: the dimension sweep")
    parser.add_argument("--cache-dir", default=None,
                        help="where sift-128-euclidean.hdf5 lives (default: <data-dir>)")
    parser.add_argument("--data-dir", default=None,
                        help="where the arXiv-CS vectors live (default: emeans/runs/data)")


def out_dir(args):
    return Path(getattr(args, "out", None) or EXPERIMENT_DIR)


def tags():
    """Which tags already have results, for `cli list`."""
    results = EXPERIMENT_DIR / "results"
    return sorted(p.name for p in results.iterdir() if p.is_dir()) if results.is_dir() else []


def run(args):
    scale = scale_for(args.tag)
    if args.dry_run:
        # No part builds its units yet; later slices list them here.
        print(f"part {args.part}, tag {args.tag}: k={scale.k}, no units of work built yet")
        return
    raise SystemExit("only --dry-run exists so far; the parts arrive in later slices")


def report(args):
    """Tables and figures are written from results/ in a later slice."""


def dataset():
    raise SystemExit("emeans/DATASET.md is filled once the arXiv-CS vectors exist")

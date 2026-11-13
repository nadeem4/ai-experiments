"""How `cli` runs the emeans experiment.

The contract `cli` needs, the two scales, `--dry-run`, and part A. Parts B and
C arrive in later slices.
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


SIFT_FILE = "sift-128-euclidean.hdf5"
SIFT_TRAIN = 900_000    # 1M base rows minus the 100k test set
ARXIV_TRAIN = 860_000   # about 960k docs minus the 100k test set; a guess until E3 counts them


def data_dir(args):
    return Path(args.data_dir) if args.data_dir else out_dir(args) / "runs" / "data"


def load_split(args, scale):
    from . import data

    path = Path(args.cache_dir or data_dir(args)) / SIFT_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found: download {SIFT_FILE} from ann-benchmarks.com "
                         "and point --cache-dir at its folder")
    return data.load_sift(path, limit=scale.n, n_test=scale.n_test, n_queries=scale.n_queries)


def store_path(args):
    return out_dir(args) / "runs" / args.tag / f"{args.part}.jsonl"


def part_a_units(args):
    from . import race
    from .consts import ARMS

    scale = scale_for(args.tag)
    # Budgets need the full-scale calibration, so the pilot leaves them out.
    return race.units("sift", ARMS, scale.seeds_a, with_budgets=args.tag == "full")


def projection(records):
    from . import pilot

    seeds = FULL.seeds_a
    return {"a_sift": pilot.project(records, (SIFT_TRAIN, FULL.k, 128), seeds),
            "a_arxiv": {**pilot.project(records, (ARXIV_TRAIN, FULL.k, 768), seeds),
                        "extrapolated": True},  # no arXiv vectors exist at pilot time
            "b": None, "c": None}  # filled in by the growing-index and sweep slices


def run(args):
    scale = scale_for(args.tag)
    if args.part != "a":
        if args.dry_run:
            print(f"part {args.part}, tag {args.tag}: its units are listed once the part is built")
            return
        raise SystemExit(f"part {args.part} arrives in a later slice")
    units = part_a_units(args)
    if args.dry_run:
        from . import store

        done = store.done_keys(store_path(args))
        print(f"part a, tag {args.tag}: k={scale.k}, {scale.seeds_a} seed(s), "
              f"{len(units)} units, {len([u for u in units if u in done])} already done")
        for u in units:
            print(f"  {' '.join(map(str, u))}{'  (done)' if u in done else ''}")
        return
    import json

    from . import env, race

    threads = env.physical_cores()
    env.pin_threads(threads)
    env_path = out_dir(args) / "runs" / args.tag / "environment.json"
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(json.dumps(env.environment(), indent=2), encoding="utf-8")
    race.run(units, load_split(args, scale), scale, store_path(args), threads=threads)
    if args.tag == "pilot":
        from . import store

        out = out_dir(args) / "results" / "pilot" / "projection.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        proj = projection(store.load(store_path(args)))
        out.write_text(json.dumps(proj, indent=2), encoding="utf-8")
        print(json.dumps(proj, indent=2))


def report(args):
    """Print the pilot's checks and exit non-zero if any fail. Tables and
    figures are written from results/ in a later slice."""
    if args.dry_run or args.part != "a" or not store_path(args).exists():
        return
    from . import pilot, store

    found = pilot.problems(store.load(store_path(args)),
                           scale_for(args.tag).k if args.tag == "pilot" else None)
    for message in found:
        print(f"PROBLEM: {message}")
    if found:
        raise SystemExit(1)
    print("pilot checks: no problems")


def dataset():
    raise SystemExit("emeans/DATASET.md is filled once the arXiv-CS vectors exist")

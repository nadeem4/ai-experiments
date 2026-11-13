"""How `cli` runs the emeans experiment.

The contract `cli` needs, the two scales, `--dry-run`, and parts A, B and C.
"""
from pathlib import Path
from typing import NamedTuple, Optional

from . import race
from .consts import B_NPROBES, DIMS

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
ARXIV_TRAIN = 860_000   # about 960k docs minus the 100k test set; a guess until DATASET.md has the real count


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
    from .consts import ARMS

    scale = scale_for(args.tag)
    # Budgets need the full-scale calibration, so the pilot leaves them out.
    return race.units("sift", ARMS, scale.seeds_a, with_budgets=args.tag == "full")


def units_for(args):
    """Every unit of the chosen part at the chosen scale, in run order."""
    from . import growing

    scale = scale_for(args.tag)
    if args.part == "b":
        return growing.units(scale)
    if args.part == "c":
        return race.sweep_units(scale, DIMS)
    return part_a_units(args)


def _arxiv_vectors(args):
    path = data_dir(args) / "vectors.npy"
    if not path.exists():
        raise SystemExit(f"{path} not found: run `uv run cli fetch emeans -p emeans/runs/data` "
                         "(or point --data-dir at the Kaggle output) first")
    return path


def load_growing(args, scale):
    from . import data, growing

    x, months = data.load_arxiv(_arxiv_vectors(args))
    return growing.make_split(x, months, scale.b_months)


def load_static(args, scale):
    from . import data

    x, _ = data.load_arxiv(_arxiv_vectors(args))
    return data.make_static_split(x, n_test=scale.n_test, n_train=scale.n)


def projection(records, b_records=()):
    """Hours per part from whatever pilot records exist; a part with no pilot
    records yet is None. C is projected from A's per-epoch rate."""
    from . import pilot

    seeds = FULL.seeds_a
    out = {"a_sift": None, "a_arxiv": None, "b": None, "c": None}
    if records:
        out["a_sift"] = pilot.project(records, (SIFT_TRAIN, FULL.k, 128), seeds)
        out["a_arxiv"] = {**pilot.project(records, (ARXIV_TRAIN, FULL.k, 768), seeds),
                          "extrapolated": True}  # no arXiv vectors exist at the A pilot
        out["c"] = {**pilot.project_c(records, ARXIV_TRAIN, FULL.k, DIMS, FULL.seeds_bc), "extrapolated": True}
    if b_records:
        sizes = {r["setting"]: r["n"] for r in b_records if r["part"] == "b"}
        months = sorted(sizes)
        per_month = (sizes[months[-1]] - sizes[months[0]]) / (len(months) - 1) if len(months) > 1 else 0
        out["b"] = pilot.project_b(b_records, FULL.b_months, sizes[months[0]] - per_month, per_month,
                                   FULL.k, 768, seeds=FULL.seeds_bc)
    return out


def write_projection(args):
    import json

    from . import store

    runs = out_dir(args) / "runs" / args.tag
    load = lambda part: store.load(runs / f"{part}.jsonl")  # noqa: E731
    out = out_dir(args) / "results" / "pilot" / "projection.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    proj = projection(load("a"), load("b"))
    out.write_text(json.dumps(proj, indent=2), encoding="utf-8")
    print(json.dumps(proj, indent=2))


def run(args):
    scale = scale_for(args.tag)
    units = units_for(args)
    if args.dry_run:
        from . import store

        done = store.done_keys(store_path(args))
        seeds = scale.seeds_a if args.part == "a" else scale.seeds_bc
        print(f"part {args.part}, tag {args.tag}: k={scale.k}, {seeds} seed(s), "
              f"{len(units)} units, {len([u for u in units if u in done])} already done")
        for u in units:
            print(f"  {' '.join(map(str, u))}{'  (done)' if u in done else ''}")
        return
    import json

    from . import env, growing

    if args.part == "a":
        split = load_split(args, scale)
    elif args.part == "b":
        split = load_growing(args, scale)
    else:
        split = load_static(args, scale)
    threads = env.physical_cores()
    env.pin_threads(threads)
    env_path = out_dir(args) / "runs" / args.tag / "environment.json"
    if not env_path.exists():
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(json.dumps(env.environment(), indent=2), encoding="utf-8")
    if args.part == "a":
        race.run(units, split, scale, store_path(args), threads=threads)
    elif args.part == "b":
        nprobes = B_NPROBES + ((scale.k,) if args.tag == "pilot" else ())  # the pilot checks nprobe = k
        growing.run(split, scale, store_path(args), nprobes=nprobes, threads=threads)
    else:
        race.run_sweep(split, scale, store_path(args), DIMS, threads=threads)
    if args.tag == "pilot":
        write_projection(args)


def write_results(args):
    """Write `results/<tag>/`: summary.json, the CSVs and `figures/`, from every
    part's records under `runs/<tag>/`. The figures then read summary.json alone."""
    import json

    from . import figures, store, tables

    runs = out_dir(args) / "runs" / args.tag
    results = out_dir(args) / "results" / args.tag
    existing = results / "summary.json"
    summary = json.loads(existing.read_text(encoding="utf-8")) if existing.exists() else {}
    for part in "abc":  # each part on its own: an incomplete one never blocks a finished one
        try:
            summary.update(tables.summary(store.load(runs / f"{part}.jsonl")))
        except ValueError as e:
            print(f"part {part}: not summarised: {e}")
    if not summary:
        raise SystemExit(f"nothing to summarise for {args.tag}")
    tables.write_summary(summary, results)
    tables.write_csv(summary, results)
    written = figures.write_all(json.loads((results / "summary.json").read_text(encoding="utf-8")),
                                results / "figures")
    for name, where in written.items():
        print(f"figure {name}: {where}")


def report(args):
    """Print the pilot's checks and exit non-zero if any fail; part C also
    prints epochs to match and speedup per dimension. Then write the summary,
    CSVs and figures under `results/<tag>/` (see `write_results`)."""
    if args.dry_run or not store_path(args).exists():
        return
    from . import growing, pilot, store, tables

    records = store.load(store_path(args))
    pilot_k = scale_for(args.tag).k if args.tag == "pilot" else None
    if args.part == "b":
        found = pilot.problems_b(records, pilot_k)
    else:
        found = pilot.problems(records, pilot_k if args.part == "a" else None)
    for message in found:
        print(f"PROBLEM: {message}")
    if found:
        raise SystemExit(1)
    print("pilot checks: no problems")
    if args.part == "c":
        for d in DIMS:
            name = f"arxiv-d{d}"
            if not any(r["dataset"] == name for r in records):
                continue
            e, s = tables.epochs_to_match(records, name, "emeans"), tables.speedup(records, name, "emeans")
            print(f"{name}: epochs to match {e if e else 'did not match within 5 epochs'}, "
                  f"speedup {f'{s:.2f}x' if s else 'none'}")
    write_results(args)


def dataset(directory=None):
    """The card `uv run cli dataset emeans` renders, from the fetched Kaggle output."""
    import json

    from . import data

    directory = Path(directory) if directory else EXPERIMENT_DIR / "runs" / "data"
    if not (directory / "vectors.npy").exists():
        raise SystemExit(f"no vectors.npy in {directory}: run `uv run cli submit emeans` "
                         "and `uv run cli fetch emeans -p emeans/runs/data` first")
    _, months = data.load_arxiv(directory / "vectors.npy")
    meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
    versions = meta.get("versions", {})
    counts = {m: int((months == m).sum()) for m in sorted(set(months.tolist()))}
    return [{
        "name": "arXiv-CS abstracts",
        "what_it_is": "Computer-science arXiv abstracts, embedded once on a Kaggle GPU "
                      "(not timed), float32 and unit-normalised, with the publish month "
                      "stored. Truncation to smaller dimensions is plain truncation then "
                      "re-normalisation, not the model card's layer-norm recipe.",
        "source": "Cornell University arXiv metadata (Kaggle)",
        "url": "https://www.kaggle.com/datasets/Cornell-University/arxiv",
        "licence": "CC0",
        "splits": {"docs": len(months)},
        "used": "cs.* primary category, 2018-01 to 2025-12, uniform sample of up to "
                "10,000 abstracts per month (seed 0); fixed seed-0 split: 100k held-out "
                "test set, 10k queries drawn from it and never indexed or trained on",
        "rows": [{"id": "see ids.txt", "month": next(iter(counts)), "vector": "768 floats"}],
        "extra": {"Model": meta["model"],
                  "Model revision": f"{meta['revision']} (prefix `search_document: `)",
                  "Dimensions": meta["dimensions"],
                  "Versions": ", ".join(f"{k} {v}" for k, v in versions.items()
                                        if k != "remote_code_revision"),
                  "Remote code revision": versions.get("remote_code_revision"),
                  "Sampling seed": 0,
                  "Per-month counts": ", ".join(f"{m}: {n}" for m, n in counts.items())},
    }]

"""Part B: a vector index that grows month by month, and four ways to keep its
coarse quantiser fresh. Timeline and held-out queries first."""
import os
import time
import zlib
from pathlib import Path
from typing import NamedTuple

import numpy as np

from . import arms, data, index, metrics, race, store
from .consts import B_NPROBES

INITIAL = ("2018-01", "2019-12")  # the initial index: these months, all docs
FIRST_ARRIVAL = 2020


def timeline(n_months=72):
    """The arrival months, 2020-01 onwards (72 at full scale, 3 for the pilot)."""
    return [f"{FIRST_ARRIVAL + i // 12}-{i % 12 + 1:02d}" for i in range(n_months)]


def hold_out(month_ids, n=500, dataset_seed=0):
    """-> (queries_idx, indexed_idx), row ids into the docs `month_ids` labels.

    Every arrival month gives up `n` docs as queries, drawn by a generator seeded
    from the dataset seed and the month alone, so every arm and every run seed
    sees the same queries. `indexed_idx` is everything else, in chronological
    order (stable), so the index as of any month is a prefix of it."""
    month_ids = np.asarray(month_ids)
    queries = []
    for month in sorted(set(month_ids.tolist())):
        if month <= INITIAL[1]:
            continue
        rows = np.flatnonzero(month_ids == month)
        if len(rows) < n:
            raise ValueError(f"month {month} has {len(rows)} docs, fewer than the {n} held out as queries")
        rng = np.random.default_rng([dataset_seed, int(month.replace("-", ""))])
        queries.append(np.sort(rng.choice(rows, n, replace=False)))
    queries = np.concatenate(queries) if queries else np.array([], dtype=np.int64)
    keep = np.ones(len(month_ids), dtype=bool)
    keep[queries] = False
    rest = np.flatnonzero(keep)
    indexed = rest[np.argsort(month_ids[rest], kind="stable")]
    return queries, indexed


def n_through(indexed_months, month):
    """How many of the chronological indexed docs exist once `month` has
    arrived: the initial docs plus every non-query doc up to and including it.
    The index at `month` is `vectors[:n_through(...)]`, a view, never a copy."""
    return int(np.searchsorted(indexed_months, month, side="right"))


class GrowingSplit(NamedTuple):
    vectors: np.ndarray   # float32, indexed docs in chronological order
    months: np.ndarray    # month label per row of `vectors`
    queries: dict         # month -> float32 (n_queries, d), never indexed or trained on
    timeline: list


def make_split(vectors, months, n_months, n_queries=500, dataset_seed=0):
    """Docs from the initial period and the first `n_months` arrival months,
    queries moved out, the rest sorted chronologically."""
    tl = timeline(n_months)
    months = np.asarray(months)
    use = np.flatnonzero((months >= INITIAL[0]) & (months <= tl[-1]))
    vectors, months = vectors[use], months[use]
    q_idx, idx = hold_out(months, n_queries, dataset_seed)
    queries = {m: np.ascontiguousarray(vectors[q_idx[months[q_idx] == m]], dtype=np.float32) for m in tl}
    return GrowingSplit(np.ascontiguousarray(vectors[idx], dtype=np.float32), months[idx], queries, tl)


ARMS_B = ("stale", "yearly", "emeans", "lloyd_half")


def seed_for(seed, key):
    """A stable integer seed from the run seed and a month or year label."""
    return zlib.crc32(f"{seed}:{key}".encode()) % 2**31


def initial_centroids(x_init, k, seed):
    """Lloyd-25 from a seeded random init on the initial index's docs."""
    return arms.train_lloyd(x_init, k, 25, arms.random_init(x_init, k, seed)).centroids


class Maintained(NamedTuple):
    centroids: np.ndarray
    train_seconds: float
    assign_seconds: float
    labels: np.ndarray   # nearest-centroid label of every indexed doc (our own, never em.assignments)
    rows: int            # rows the training touched


def maintain(arm, centroids, x_indexed, month, seed, labels_prev):
    """One month's maintenance of the quantiser; k comes from `centroids`.

    Stale (and yearly outside January) trains nothing: the new docs are assigned
    untimed and maintenance is 0. Otherwise maintenance is training plus
    re-assignment of every indexed doc, both timed; adding docs to the index is
    not part of it. Labels always come from our re-assignment: a half-epoch
    e-means run leaves some of its own assignments at -1 by design."""
    k = len(centroids)
    trained = None
    if arm == "yearly" and month.endswith("-01"):
        year = month[:4]
        trained = arms.train_lloyd(x_indexed, k, 25, arms.random_init(x_indexed, k, seed_for(seed, year)))
    elif arm == "emeans":
        trained = arms.train_emeans(x_indexed, k, 0.5, centroids, seed_for(seed, month))
    elif arm == "lloyd_half":
        rng = np.random.default_rng(seed_for(seed, month))
        start = time.perf_counter()
        half = x_indexed[rng.choice(len(x_indexed), len(x_indexed) // 2, replace=False)]
        gather = time.perf_counter() - start  # e-means shuffles inside its timer, so this is timed too
        one = arms.train_lloyd(half, k, 1, centroids)
        trained = arms.Trained(one.centroids, one.seconds + gather, len(half))
    elif arm not in ARMS_B:
        raise ValueError(f"unknown arm {arm!r}")
    if trained is None:  # stale
        new = x_indexed[len(labels_prev):]
        fresh = index.reassign(new, centroids)[0] if len(new) else np.array([], dtype=np.int64)
        return Maintained(centroids, 0.0, 0.0, np.concatenate([labels_prev, fresh]), 0)
    labels, assign_seconds = index.reassign(x_indexed, trained.centroids)
    return Maintained(trained.centroids, trained.seconds, assign_seconds, labels, trained.rows)


REQUIRED = race.CORE + ("n_queries", "centroids_hash", "train_seconds", "assign_seconds", "maintenance_seconds",
                        "adding_seconds", "eval_seconds", "rows", "imbalance", "dead_clusters", "search",
                        "recall_per_query")


def units(scale, seeds=None, arm_names=ARMS_B, tl=None):
    """Every unit in run order: chains (seed, arm) outer, months inner."""
    tl = (tl or timeline(scale.b_months))[:scale.b_months]
    return [("b", "arxiv", arm, month, seed)
            for seed in (range(scale.seeds_bc) if seeds is None else seeds)
            for arm in arm_names for month in tl]


def _save_atomic(path, **arrays):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _truth(gs, run_dir, month, n):
    """Exact top-10 ids into the indexed prefix, cached once per month for every arm and seed."""
    path = run_dir / "truth" / f"{month}.npy"
    if path.exists():
        return np.load(path)
    truth = data.exact_neighbours(gs.vectors[:n], gs.queries[month], 10)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(month + ".tmp.npy")
    np.save(tmp, truth)
    os.replace(tmp, path)
    return truth


def _initial(gs, run_dir, k, seed):
    """The shared starting centroids for a seed, computed once and cached."""
    path = run_dir / "state" / f"init-{seed}.npy"
    n0 = n_through(gs.months, INITIAL[1])
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"init-{seed}.tmp.npy")
        np.save(tmp, initial_centroids(gs.vectors[:n0], k, seed))
        os.replace(tmp, path)
    centroids = np.load(path)
    return centroids, index.reassign(gs.vectors[:n0], centroids)[0]


def _resume(run_dir, arm, seed, last):
    """-> (centroids, labels) from the saved state of the chain's last record."""
    path = run_dir / "state" / f"{arm}-{seed}-{last['setting']}.npz"
    try:
        with np.load(path) as z:
            centroids, labels = z["centroids"], z["labels"]
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"cannot resume chain {arm} seed {seed}: state {path} is unreadable ({e})")
    if race._hash(centroids) != last["centroids_hash"]:
        raise SystemExit(f"cannot resume chain {arm} seed {seed}: state {path} does not match "
                         f"the last record ({last['setting']})")
    return centroids, labels


def run(gs, scale, store_path, seeds=None, arm_names=ARMS_B, nprobes=B_NPROBES, threads=None, log=print):
    """Run every month of every (seed, arm) chain not already recorded.

    Per month: maintain, build the IVF over the indexed prefix (the add step,
    timed apart), evaluate the month's queries against cached exact truth, then
    save the chain state (atomically) and append the record. A crash between the
    two leaves the state a month ahead; the previous month's state is kept, so
    that month is simply replayed."""
    from . import env

    threads = threads or env.physical_cores()
    env.pin_threads(threads)
    store_path = Path(store_path)
    run_dir = store_path.parent
    tl = gs.timeline[:scale.b_months]
    for seed in range(scale.seeds_bc) if seeds is None else seeds:
        for arm in arm_names:
            records = [r for r in store.load(store_path)
                       if r["part"] == "b" and r["arm"] == arm and r["seed"] == seed]
            done = {r["setting"]: r for r in records}
            if set(done) != set(tl[:len(done)]):
                raise SystemExit(f"chain {arm} seed {seed}: recorded months are not a prefix of the timeline")
            if len(done) == len(tl):
                continue
            if done:
                centroids, labels = _resume(run_dir, arm, seed, done[tl[len(done) - 1]])
            else:
                centroids, labels = _initial(gs, run_dir, scale.k, seed)
            for i in range(len(done), len(tl)):
                month = tl[i]
                n = n_through(gs.months, month)
                x = gs.vectors[:n]
                m = maintain(arm, centroids, x, month, seed, labels)
                centroids, labels = m.centroids, m.labels
                k = len(centroids)
                start = time.perf_counter()
                ivf = index.build_ivf(centroids, x, labels)
                adding = time.perf_counter() - start
                start = time.perf_counter()
                queries = gs.queries[month]
                summary, per_query = index.evaluate(ivf, queries, _truth(gs, run_dir, month, n), nprobes, base=x)
                evaluation = time.perf_counter() - start
                record = {"part": "b", "dataset": "arxiv", "arm": arm, "setting": month, "seed": seed,
                          "n": n, "k": k, "d": x.shape[1], "threads": threads,
                          "observed_threads": race._observed_threads(threads), "n_queries": len(queries),
                          "centroids_hash": race._hash(centroids), "train_seconds": m.train_seconds,
                          "assign_seconds": m.assign_seconds,
                          "maintenance_seconds": m.train_seconds + m.assign_seconds,
                          "adding_seconds": adding, "eval_seconds": evaluation, "rows": m.rows,
                          "imbalance": metrics.imbalance(labels, k),
                          "dead_clusters": metrics.dead_clusters(labels, k),
                          "search": {str(p): v for p, v in summary.items()},
                          "recall_per_query": {str(p): v.tolist() for p, v in per_query.items()}}
                (run_dir / "state").mkdir(parents=True, exist_ok=True)
                _save_atomic(run_dir / "state" / f"{arm}-{seed}-{month}.npz", centroids=centroids, labels=labels)
                store.append(store_path, record)
                if i >= 2:  # keep the previous month's state for a replay
                    (run_dir / "state" / f"{arm}-{seed}-{tl[i - 2]}.npz").unlink(missing_ok=True)
                log(f"b {arm} seed {seed} {month}")

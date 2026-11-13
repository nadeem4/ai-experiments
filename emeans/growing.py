"""Part B: a vector index that grows month by month, and four ways to keep its
coarse quantiser fresh. Timeline and held-out queries first."""
from typing import NamedTuple

import numpy as np

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

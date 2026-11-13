"""Datasets and the fixed seed-0 split.

One split per dataset, never re-drawn: 100k vectors are held out as the MSE test
set and the rest train. Queries never appear in an index or a training set.
"""
from pathlib import Path
from typing import NamedTuple

import faiss
import numpy as np


class Split(NamedTuple):
    train: np.ndarray    # float32 (n_train, d)
    test: np.ndarray     # float32 (n_test, d), the MSE test set
    index: np.ndarray    # float32, what the IVF index holds
    queries: np.ndarray  # float32 (nq, d)
    truth: np.ndarray    # int64 (nq, 10), indexes into `index`


def exact_neighbours(index, queries, k=10):
    """Exact top-k by L2, as int64 ids into `index`."""
    index = np.ascontiguousarray(index, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    flat = faiss.IndexFlatL2(index.shape[1])
    flat.add(index)
    return flat.search(queries, k)[1].astype(np.int64)


def make_split(x, seed=0, n_test=100_000, queries=None, truth=None, n_queries=10_000):
    """Split `x` into train and test. With `queries` and `truth` (SIFT) the
    index is all of `x` and the shipped truth stays valid. Without them (arXiv)
    the queries are `n_queries` rows of the test set, removed from the index,
    and the truth is computed here."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    if n_test >= len(x) or n_queries > n_test:
        raise ValueError(f"cannot split {len(x)} rows with n_test={n_test}, n_queries={n_queries}")
    perm = np.random.default_rng(seed).permutation(len(x))
    test_ids, train_ids = perm[:n_test], perm[n_test:]
    train, test = x[train_ids], x[test_ids]
    if queries is not None:
        q = np.ascontiguousarray(queries[:n_queries], dtype=np.float32)
        return Split(train, test, x, q, np.asarray(truth)[:n_queries].astype(np.int64))
    query_ids = test_ids[:n_queries]
    keep = np.ones(len(x), dtype=bool)
    keep[query_ids] = False
    index, q = x[keep], x[query_ids]
    return Split(train, test, index, q, exact_neighbours(index, q, 10))


def load_sift(path, limit=None, seed=0, n_test=100_000, n_queries=10_000):
    """ann-benchmarks `sift-128-euclidean.hdf5` (keys `train`, `test`, `neighbors`).

    With `limit` only the first `limit` base rows are used and the truth is
    recomputed, because the shipped truth only holds for the full base."""
    import h5py

    with h5py.File(Path(path), "r") as f:
        base = np.asarray(f["train"], dtype=np.float32)
        queries = np.asarray(f["test"], dtype=np.float32)
        shipped = np.asarray(f["neighbors"])
    if limit is not None:
        base = base[:limit]
        truth = exact_neighbours(base, queries[:n_queries], 10)
    else:
        truth = shipped[:, :10].astype(np.int64)
    return make_split(base, seed=seed, n_test=n_test, queries=queries, truth=truth,
                      n_queries=n_queries)


def load_arxiv(path):
    """`vectors.npy` from the Kaggle run (months.npy beside it) -> (vectors, months)."""
    path = Path(path)
    return (np.load(path).astype(np.float32, copy=False),
            np.load(path.with_name("months.npy")))


def truncate(x, d):
    """First `d` columns, re-normalised (plain truncation, not the model card's
    layer-norm recipe)."""
    if d >= x.shape[1]:
        return x
    t = np.ascontiguousarray(x[:, :d], dtype=np.float32)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def month_order(months):
    """Stable argsort: documents arrive month by month, ties in file order."""
    return np.argsort(months, kind="stable")

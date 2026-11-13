"""The three clustering arms, all started from the same init per seed.

Each returns a `Trained`: centroids, the seconds spent training (perf_counter
around the training call only) and the rows it touched. Defaults everywhere
except what the protocol fixes; nothing is tuned.
"""
import time
from typing import NamedTuple

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans
from sklearn.cluster import MiniBatchKMeans

from .consts import LLOYD_AT


class Trained(NamedTuple):
    centroids: np.ndarray  # float32 (k, d)
    seconds: float
    rows: int              # rows touched by training


def random_init(x, k, seed):
    """k distinct rows of x chosen by RandomState(seed); the same init feeds
    every arm, so they are paired per seed."""
    ids = np.random.RandomState(seed).choice(len(x), k, replace=False)
    return np.ascontiguousarray(x[ids], dtype=np.float32)


def _f32(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def train_lloyd(x, k, iters, init):
    x, init = _f32(x), _f32(init)
    km = faiss.Kmeans(x.shape[1], k, niter=iters, nredo=1, verbose=False,
                      max_points_per_centroid=len(x) // k + 1)  # no subsampling
    start = time.perf_counter()
    km.train(x, init_centroids=init)
    seconds = time.perf_counter() - start
    return Trained(km.centroids.astype(np.float32), seconds, iters * len(x))


def lloyd_snapshots(x, k, init, at=LLOYD_AT):
    """Centroids after each iteration count in `at`, as a warm-started chain
    (faiss has no per-iteration hook). Untimed; see the README for why this is
    an approximation of one uninterrupted run."""
    out, done, current = {}, 0, init
    for n in sorted(at):
        current = train_lloyd(x, k, n - done, current).centroids
        out[n], done = current, n
    return out


def train_emeans(x, k, epochs, init, seed):
    x, init = _f32(x), _f32(init)
    em = NumpyEmeans(x.shape[1], k, nepochs=epochs, seed=seed)
    start = time.perf_counter()
    centroids, assignments = em.train(x, init_centroids=init)
    seconds = time.perf_counter() - start
    if epochs >= 1 and (assignments < 0).any():
        raise AssertionError("e-means left rows unassigned over whole epochs")
    return Trained(np.asarray(centroids, dtype=np.float32), seconds,
                   int(sum(em.recipe["epoch_rows"])))


def train_minibatch(x, k, epochs, init, seed):
    x, init = _f32(x), _f32(init)
    km = MiniBatchKMeans(k, init=init, n_init=1, max_iter=epochs, max_no_improvement=None,
                         tol=0, random_state=seed, compute_labels=False)
    start = time.perf_counter()
    km.fit(x)
    seconds = time.perf_counter() - start
    return Trained(km.cluster_centers_.astype(np.float32), seconds,
                   int(km.n_steps_ * km.batch_size))


def train(name, x, k, epochs, init, seed):
    """One arm by name; for Lloyd `epochs` counts iterations."""
    if name == "lloyd":
        return train_lloyd(x, k, epochs, init)
    if name == "emeans":
        return train_emeans(x, k, epochs, init, seed)
    if name == "minibatch":
        return train_minibatch(x, k, epochs, init, seed)
    raise ValueError(f"unknown arm {name!r}")

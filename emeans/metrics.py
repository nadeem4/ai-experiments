"""Quality and search metrics for a set of centroids.

`mean_ci` is a plain t interval, copied from `rerank` rather than imported so
that each experiment stays self-contained.
"""
import math
import statistics

import faiss
import numpy as np

Z = 1.96  # 95%
# Two-sided 95% t critical values for small samples (df = n - 1); 1.96 beyond 30.
T95 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def mean_ci(values):
    """Mean with a 95% t-interval. One value has no interval."""
    mean = statistics.fmean(values)
    if len(values) < 2:
        return mean, None, None
    t = T95[len(values) - 2] if len(values) - 1 <= len(T95) else Z
    half = t * statistics.stdev(values) / math.sqrt(len(values))
    return mean, mean - half, mean + half


def mse(x, centroids):
    """Mean squared L2 distance from each row of `x` to its nearest centroid."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    distances, _ = faiss.knn(x, centroids, 1)
    return float(distances.astype(np.float64).mean())


def _counts(labels, k):
    return np.bincount(np.asarray(labels), minlength=k).astype(np.float64)


def imbalance(labels, k):
    """k * sum(n_i^2) / n^2: 1.0 is perfectly even, k is everything in one list."""
    counts = _counts(labels, k)
    return float(k * (counts ** 2).sum() / counts.sum() ** 2)


def dead_clusters(labels, k, threshold=10):
    """Clusters with fewer than `threshold` points, empty ones included."""
    return int((_counts(labels, k) < threshold).sum())


def recall_at_k(found, truth, k=10):
    """-> (mean, per-query array): the share of the true top-k that was found."""
    found, truth = np.asarray(found)[:, :k], np.asarray(truth)[:, :k]
    per_query = (found[:, :, None] == truth[:, None, :]).any(axis=1).sum(axis=1) / k
    return float(per_query.mean()), per_query


def scanned(index, queries, nprobe):
    """Vectors scanned per query: the summed sizes of the `nprobe` lists probed."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    sizes = np.array([index.invlists.list_size(i) for i in range(index.nlist)], dtype=np.int64)
    _, probes = index.quantizer.search(queries, nprobe)
    return sizes[probes].sum(axis=1)


def summarise_scanned(values):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(np.percentile(values, 95))

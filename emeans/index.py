"""IVF indexes built from a given set of centroids, and how well they search."""
import time

import faiss
import numpy as np

from . import metrics


def reassign(x, centroids):
    """-> (nearest-centroid label per row as int64, seconds). Timed: in the
    growing index this is part of the maintenance cost."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    start = time.perf_counter()
    _, labels = faiss.knn(x, centroids, 1)
    seconds = time.perf_counter() - start
    return labels[:, 0].astype(np.int64), seconds


def build_ivf(centroids, x, labels):
    """An IndexIVFFlat whose coarse quantiser is `centroids` as-is, with every
    row of `x` added to the list given by `labels`."""
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    k, d = centroids.shape
    quantizer = faiss.IndexFlatL2(d)
    quantizer.add(centroids)
    ivf = faiss.IndexIVFFlat(quantizer, d, k)
    ivf.is_trained = True  # the centroids are the training
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    ids = np.arange(len(x), dtype=np.int64)
    ivf.add_core(len(x), faiss.swig_ptr(x), faiss.swig_ptr(ids), faiss.swig_ptr(labels))
    return ivf


def search(ivf, queries, nprobe, k=10):
    """-> (ids (nq, k), queries per second). One warm-up pass, then one timed
    batch search."""
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    ivf.nprobe = nprobe
    ivf.search(queries, k)
    start = time.perf_counter()
    _, ids = ivf.search(queries, k)
    seconds = time.perf_counter() - start
    return ids, len(queries) / seconds


def evaluate(ivf, queries, truth, nprobes):
    """-> (summary, per_query): `summary[nprobe]` holds recall10, scanned_mean,
    scanned_p95 and qps; `per_query[nprobe]` is the per-query recall array.
    Keys are ints here; a record stores them as strings."""
    summary, per_query = {}, {}
    for nprobe in nprobes:
        ids, qps = search(ivf, queries, nprobe)
        recall, per = metrics.recall_at_k(ids, truth)
        mean, p95 = metrics.summarise_scanned(metrics.scanned(ivf, queries, nprobe))
        summary[nprobe] = {"recall10": recall, "scanned_mean": mean, "scanned_p95": p95, "qps": qps}
        per_query[nprobe] = per
    return summary, per_query

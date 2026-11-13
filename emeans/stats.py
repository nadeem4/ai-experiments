"""The paired bootstrap behind part B's end-state claims."""
import numpy as np


def paired_bootstrap(a, b, n_boot=10_000, seed=0):
    """-> (mean_diff, lo, hi): the mean of `a - b` per query and its 95% interval,
    from resampling queries with their pairing kept (the same indices for both)."""
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    chunk = 500
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        means[start:start + size] = diff[rng.integers(0, len(diff), (size, len(diff)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)

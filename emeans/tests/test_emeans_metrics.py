"""MSE, imbalance, dead clusters, recall and scan counts."""
import faiss
import numpy as np
import pytest

from emeans import metrics


def test_mse_is_zero_on_the_centroids_and_exact_on_a_known_case():
    c = np.array([[0, 0], [10, 0]], dtype=np.float32)
    assert metrics.mse(c.copy(), c) == 0.0
    x = np.array([[0, 0], [2, 0]], dtype=np.float32)
    assert metrics.mse(x, np.array([[1, 0]], dtype=np.float32)) == pytest.approx(1.0)


def test_imbalance_is_one_when_even_and_k_when_all_in_one():
    assert metrics.imbalance(np.repeat(np.arange(4), 5), 4) == pytest.approx(1.0)
    assert metrics.imbalance(np.zeros(20, dtype=np.int64), 4) == pytest.approx(4.0)


def test_dead_clusters_counts_small_and_empty_ones():
    labels = np.concatenate([np.zeros(50), np.ones(9), np.full(10, 2)]).astype(np.int64)
    # cluster 1 has 9 (dead), cluster 2 has exactly 10 (alive), 3 and 4 are empty
    assert metrics.dead_clusters(labels, 5) == 3
    assert metrics.dead_clusters(labels, 5, threshold=1) == 2


def test_recall_identical_half_and_shape():
    truth = np.arange(20).reshape(2, 10)
    r, per = metrics.recall_at_k(truth.copy(), truth)
    assert r == 1.0 and per.shape == (2,)
    half = truth.copy()
    half[:, 5:] = -1
    r, per = metrics.recall_at_k(half, truth)
    assert r == pytest.approx(0.5) and (per == 0.5).all()


def small_ivf():
    """Four lists of known sizes 1, 2, 3, 4 along a line."""
    centroids = np.array([[0, 0], [10, 0], [100, 0], [200, 0]], dtype=np.float32)
    pts = [[0, 1]] + [[10, 1], [10, 2]] + [[100, 1], [100, 2], [100, 3]] + \
          [[200, 1], [200, 2], [200, 3], [200, 4]]
    quantizer = faiss.IndexFlatL2(2)
    quantizer.add(centroids)
    ivf = faiss.IndexIVFFlat(quantizer, 2, 4)
    ivf.train(centroids)
    ivf.add(np.array(pts, dtype=np.float32))
    return ivf


def test_scanned_is_the_sum_of_the_probed_list_sizes():
    ivf = small_ivf()
    q = np.array([[1, 0], [160, 0]], dtype=np.float32)
    assert metrics.scanned(ivf, q, 1).tolist() == [1, 4]
    assert metrics.scanned(ivf, q, 2).tolist() == [1 + 2, 4 + 3]
    assert metrics.scanned(ivf, q, 4).tolist() == [10, 10]


def test_summarise_scanned_mean_and_p95():
    mean, p95 = metrics.summarise_scanned(np.arange(1, 101))
    assert mean == pytest.approx(50.5)
    assert p95 == pytest.approx(95.05)


def test_mean_ci_has_an_interval_only_with_two_values():
    assert metrics.mean_ci([1.0]) == (1.0, None, None)
    m, lo, hi = metrics.mean_ci([1.0, 2.0, 3.0])
    assert m == 2.0 and lo < 2.0 < hi

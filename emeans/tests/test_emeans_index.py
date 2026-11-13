"""IVF build and search, on a synthetic 3,000 x 16 set with k = 32."""
import numpy as np
import pytest

from emeans import data, index

K = 32


@pytest.fixture(scope="module")
def world():
    rng = np.random.default_rng(0)
    x = rng.random((3000, 16), dtype=np.float32)
    centroids = x[rng.choice(len(x), K, replace=False)].copy()
    labels, _ = index.reassign(x, centroids)
    ivf = index.build_ivf(centroids, x, labels)
    queries = rng.random((100, 16), dtype=np.float32)
    truth = data.exact_neighbours(x, queries, 10)
    return x, centroids, labels, ivf, queries, truth


def test_reassign_returns_nearest_labels_and_a_time(world):
    x, centroids, labels, *_ = world
    brute = ((x[:50, None, :] - centroids[None]) ** 2).sum(-1).argmin(1)
    assert labels.dtype == np.int64 and labels.shape == (len(x),)
    assert (labels[:50] == brute).all()
    assert index.reassign(x, centroids)[1] > 0


def test_every_vector_sits_in_its_nearest_centroids_list(world):
    x, _, labels, ivf, *_ = world
    assert ivf.is_trained and ivf.ntotal == len(x)
    sizes = [ivf.invlists.list_size(i) for i in range(K)]
    assert sizes == np.bincount(labels, minlength=K).tolist()


def test_nprobe_equal_to_k_is_exact(world):
    _, _, _, ivf, queries, truth = world
    summary, _ = index.evaluate(ivf, queries, truth, (K,))
    assert summary[K]["recall10"] == 1.0
    assert summary[K]["scanned_mean"] == ivf.ntotal


def test_recall_does_not_fall_as_nprobe_grows(world):
    _, _, _, ivf, queries, truth = world
    summary, per_query = index.evaluate(ivf, queries, truth, (1, 4, 16))
    r = [summary[n]["recall10"] for n in (1, 4, 16)]
    assert r == sorted(r) and r[0] < r[-1]
    assert per_query[4].shape == (len(queries),)


def test_evaluate_reports_every_field_and_a_positive_qps(world):
    _, _, _, ivf, queries, truth = world
    summary, _ = index.evaluate(ivf, queries, truth, (4,))
    assert set(summary[4]) == {"recall10", "scanned_mean", "scanned_p95", "qps"}
    assert summary[4]["qps"] > 0
    ids, qps = index.search(ivf, queries, 4)
    assert ids.shape == (len(queries), 10) and qps > 0


def test_summary_survives_json_with_string_keys(world):
    import json

    _, _, _, ivf, queries, truth = world
    summary, _ = index.evaluate(ivf, queries, truth, (1, 4))
    back = json.loads(json.dumps({str(n): v for n, v in summary.items()}))
    assert back["4"]["recall10"] == summary[4]["recall10"]


def test_build_ivf_refuses_a_label_of_minus_one(world):
    x, centroids, labels, *_ = world
    bad = labels.copy()
    bad[:10] = -1
    with pytest.raises(ValueError):
        index.build_ivf(centroids, x, bad)


def test_build_ivf_refuses_a_label_of_k(world):
    x, centroids, labels, *_ = world
    bad = labels.copy()
    bad[0] = K
    with pytest.raises(ValueError):
        index.build_ivf(centroids, x, bad)


def test_build_ivf_refuses_a_label_array_of_the_wrong_length(world):
    x, centroids, labels, *_ = world
    with pytest.raises(ValueError):
        index.build_ivf(centroids, x, labels[:-1])


def test_evaluate_with_base_is_exact_at_nprobe_k_even_with_duplicate_vectors():
    rng = np.random.default_rng(1)
    x = np.repeat(rng.random((100, 4), dtype=np.float32), 20, axis=0)  # 20 copies each
    centroids = x[rng.choice(len(x), 8, replace=False)].copy()
    labels, _ = index.reassign(x, centroids)
    ivf = index.build_ivf(centroids, x, labels)
    queries = x[::97][:20].copy()
    truth = data.exact_neighbours(x, queries, 10)
    summary, per_query = index.evaluate(ivf, queries, truth, (8,), base=x)
    assert summary[8]["recall10"] == 1.0 and (per_query[8] == 1.0).all()

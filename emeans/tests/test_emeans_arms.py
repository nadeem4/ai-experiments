"""The three arms on 4 well-separated gaussian blobs (k = 4), one thread."""
import numpy as np
import pytest

from emeans import arms, env, metrics
from emeans.consts import LLOYD_AT

K = 4


@pytest.fixture(autouse=True)
def one_thread():
    before = env.faiss.omp_get_max_threads()
    env.pin_threads(1)
    yield
    env.pin_threads(before)


@pytest.fixture(scope="module")
def blobs():
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0], [20, 0], [0, 20], [20, 20]], dtype=np.float32)
    x = (centres[rng.integers(0, K, 2000)] + rng.normal(0, 1, (2000, 2))).astype(np.float32)
    return x, arms.random_init(x, K, seed=0)


def test_random_init_is_seeded_rows_of_x(blobs):
    x, init = blobs
    assert init.shape == (K, 2) and init.dtype == np.float32
    assert np.array_equal(init, arms.random_init(x, K, seed=0))
    assert not np.array_equal(init, arms.random_init(x, K, seed=1))
    assert all((x == row).all(axis=1).any() for row in init)


def test_lloyd_train_mse_never_rises_across_snapshots(blobs):
    x, init = blobs
    snaps = arms.lloyd_snapshots(x, K, init)
    assert sorted(snaps) == sorted(LLOYD_AT)
    mses = [metrics.mse(x, snaps[i]) for i in LLOYD_AT]
    assert all(b <= a + 1e-6 for a, b in zip(mses, mses[1:]))


@pytest.mark.parametrize("name", ["lloyd", "emeans", "minibatch"])
def test_each_arm_returns_finite_float32_centroids_and_time(blobs, name):
    x, init = blobs
    t = arms.train(name, x, K, 2, init, seed=0)
    assert t.centroids.shape == (K, 2) and t.centroids.dtype == np.float32
    assert np.isfinite(t.centroids).all() and t.seconds > 0 and t.rows > 0


def test_lloyd_rows_are_iters_times_n(blobs):
    x, init = blobs
    assert arms.train_lloyd(x, K, 3, init).rows == 3 * len(x)


@pytest.mark.parametrize("name", ["emeans", "minibatch"])
def test_same_seed_same_centroids(blobs, name):
    x, init = blobs
    a = arms.train(name, x, K, 2, init, seed=3).centroids
    b = arms.train(name, x, K, 2, init, seed=3).centroids
    assert np.array_equal(a, b)


@pytest.mark.parametrize("name", ["lloyd", "emeans", "minibatch"])
def test_each_arm_recovers_separated_blobs(blobs, name):
    x, init = blobs
    # an init of four rows can miss a blob, so start from one row per blob
    good = np.array([x[np.argmin(((x - c) ** 2).sum(1))] for c in
                     ([0, 0], [20, 0], [0, 20], [20, 20])], dtype=np.float32)
    t = arms.train(name, x, K, 5, good, seed=0)
    assert metrics.mse(x, t.centroids) < 4.0  # within-blob variance is 2


def test_emeans_does_not_override_max_points_per_centroid(blobs, monkeypatch):
    x, init = blobs
    seen = {}
    real = arms.NumpyEmeans

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(arms, "NumpyEmeans", spy)
    arms.train_emeans(x, K, 1, init, seed=0)
    assert "max_points_per_centroid" not in seen

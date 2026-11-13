"""Thread pinning and the environment record."""
import faiss
import pytest
from threadpoolctl import threadpool_info

from emeans import env


@pytest.fixture(autouse=True)
def restore_threads():
    before = faiss.omp_get_max_threads()
    yield
    env.pin_threads(before)


def test_physical_cores_is_a_positive_int():
    assert isinstance(env.physical_cores(), int) and env.physical_cores() >= 1


def test_pin_threads_sets_faiss_and_every_pool():
    env.pin_threads(2)
    assert faiss.omp_get_max_threads() == 2
    assert all(p["num_threads"] <= 2 for p in threadpool_info())


def test_environment_has_the_recorded_fields():
    e = env.environment()
    for key in ("faiss", "scikit-learn", "numpy", "sentence-transformers",
                "cpu", "threads", "commit", "date"):
        assert key in e


def test_torch_faiss_and_sklearn_share_a_process():
    """The duplicate-OpenMP hazard on Windows: all three, then a pinned run."""
    import numpy as np
    import sklearn.cluster
    import torch  # noqa: F401

    env.pin_threads(1)
    x = np.random.default_rng(0).random((200, 4), dtype="float32")
    km = faiss.Kmeans(4, 4, niter=2, seed=0)
    km.train(x)
    sklearn.cluster.MiniBatchKMeans(4, n_init=1, random_state=0).fit(x)

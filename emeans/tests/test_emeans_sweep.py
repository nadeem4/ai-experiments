"""Part C on synthetic 2,000 x 64 unit vectors, dims (8, 16), k = 8, one thread."""
import numpy as np
import pytest

from emeans import data, env, race, store
from emeans.experiment import Scale

SCALE = Scale(n=None, n_test=300, n_queries=0, k=8, seeds_a=1, seeds_bc=1, b_months=3, nprobes=(1,))


@pytest.fixture(autouse=True)
def one_thread():
    before = env.faiss.omp_get_max_threads()
    env.pin_threads(1)
    yield
    env.pin_threads(before)


@pytest.fixture(scope="module")
def split():
    rng = np.random.default_rng(0)
    centres = rng.normal(size=(8, 64))
    x = centres[rng.integers(0, 8, 2000)] + rng.normal(0, 0.3, (2000, 64))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return data.make_static_split(x, n_test=300)


def test_static_split_has_no_queries_and_is_a_shuffled_train_test_split(split):
    assert len(split.train) == 1700 and len(split.test) == 300 and len(split.queries) == 0
    assert split.index is split.train


def test_vectors_are_renormalised_at_each_dimension(split):
    for d in (8, 16):
        t = race.truncated_split(split, d)
        for part in (t.train, t.test, t.index):
            assert part.shape[1] == d
            assert np.abs(np.linalg.norm(part, axis=1) - 1).max() < 1e-5


def test_the_sweep_records_both_dims_with_lloyd_and_emeans_only(split, tmp_path):
    path = tmp_path / "c.jsonl"
    race.run_sweep(split, SCALE, path, dims=(8, 16), threads=1, log=lambda *_: None)
    records = store.load(path)
    assert {r["dataset"] for r in records} == {"arxiv-d8", "arxiv-d16"}
    assert {r["arm"] for r in records} == {"lloyd", "emeans"}
    assert {r["part"] for r in records} == {"c"}
    assert not [r for r in records if r["setting"].startswith("budget=")]
    assert all("search" not in r for r in records)
    assert {r["d"] for r in records} == {8, 16}
    iters = [r for r in records if r["setting"] == "iters=25"]
    assert len(iters) == 2 and all("test_mse" in r for r in iters)


def test_sweep_resumes_without_rerunning(split, tmp_path):
    path = tmp_path / "c.jsonl"
    race.run_sweep(split, SCALE, path, dims=(8,), threads=1, log=lambda *_: None)
    before = path.read_bytes()
    race.run_sweep(split, SCALE, path, dims=(8,), threads=1, log=lambda *_: None)
    assert path.read_bytes() == before

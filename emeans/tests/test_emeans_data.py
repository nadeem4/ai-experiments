"""Splits and the SIFT loader, on synthetic arrays only."""
import h5py
import numpy as np
import pytest

from emeans import data


def make_x(n=2000, d=8, seed=0):
    """Rows whose first column is the row id, so a row can be traced by id."""
    x = np.random.default_rng(seed).random((n, d), dtype=np.float32)
    x[:, 0] = np.arange(n)
    return x


def ids(rows):
    return set(rows[:, 0].astype(int).tolist())


def test_exact_neighbours_match_brute_force():
    x = make_x()[:, 1:].copy()
    q = x[:5] + 0.01
    got = data.exact_neighbours(x, q, k=10)
    brute = np.argsort(((q[:, None, :] - x[None]) ** 2).sum(-1), axis=1)[:, :10]
    assert got.dtype == np.int64 and got.shape == (5, 10)
    assert (got == brute).all()


def test_train_and_test_are_disjoint_and_cover_everything():
    x = make_x()
    s = data.make_split(x, seed=0, n_test=300, n_queries=50)
    assert len(s.test) == 300
    assert ids(s.train).isdisjoint(ids(s.test))
    assert ids(s.train) | ids(s.test) == set(range(len(x)))


def test_same_seed_same_split_different_seed_different():
    x = make_x()
    a = data.make_split(x, seed=0, n_test=300, n_queries=50)
    b = data.make_split(x, seed=0, n_test=300, n_queries=50)
    c = data.make_split(x, seed=1, n_test=300, n_queries=50)
    assert (a.test == b.test).all() and (a.queries == b.queries).all()
    assert ids(a.test) != ids(c.test)


def test_no_query_is_in_the_index_or_the_training_set():
    x = make_x()
    s = data.make_split(x, seed=0, n_test=300, n_queries=50)
    assert len(s.queries) == 50
    assert ids(s.queries) <= ids(s.test)
    assert ids(s.queries).isdisjoint(ids(s.index))
    assert ids(s.queries).isdisjoint(ids(s.train))
    assert len(s.index) == len(x) - 50


def test_truth_is_brute_force_over_the_index():
    x = make_x()
    s = data.make_split(x, seed=0, n_test=300, n_queries=50)
    brute = np.argsort(((s.queries[:, None, :] - s.index[None]) ** 2).sum(-1), axis=1)[:, :10]
    assert s.truth.dtype == np.int64 and s.truth.shape == (50, 10)
    assert (s.truth == brute).all()


def test_given_queries_keep_the_whole_base_as_the_index():
    x = make_x()
    q = make_x(n=40, seed=5)
    truth = data.exact_neighbours(x, q, 10)
    s = data.make_split(x, seed=0, n_test=300, queries=q, truth=truth, n_queries=25)
    assert len(s.index) == len(x)
    assert len(s.queries) == 25 and len(s.truth) == 25
    assert (s.truth == truth[:25]).all()


def write_sift(path, base, queries):
    with h5py.File(path, "w") as f:
        f["train"] = base
        f["test"] = queries
        f["neighbors"] = np.tile(np.arange(100), (len(queries), 1))  # deliberately wrong


def test_load_sift_returns_float32_and_the_shipped_truth(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.random((500, 8)).astype(np.float64)
    queries = rng.random((20, 8)).astype(np.float64)
    write_sift(tmp_path / "s.hdf5", base, queries)
    s = data.load_sift(tmp_path / "s.hdf5", n_test=100, n_queries=20)
    assert s.train.dtype == np.float32 and s.index.dtype == np.float32
    assert len(s.index) == 500 and len(s.train) == 400 and len(s.test) == 100
    assert s.truth.shape == (20, 10)
    assert (s.truth == np.tile(np.arange(10), (20, 1))).all()


def test_load_sift_with_a_limit_recomputes_the_truth(tmp_path):
    rng = np.random.default_rng(0)
    base = rng.random((500, 8)).astype(np.float32)
    queries = rng.random((20, 8)).astype(np.float32)
    write_sift(tmp_path / "s.hdf5", base, queries)
    s = data.load_sift(tmp_path / "s.hdf5", limit=200, n_test=50, n_queries=10)
    assert len(s.index) == 200 and len(s.queries) == 10
    assert (s.truth == data.exact_neighbours(base[:200], queries[:10], 10)).all()
    # change one base row to sit on top of query 0: the truth must follow
    base2 = base.copy()
    base2[7] = queries[0]
    write_sift(tmp_path / "t.hdf5", base2, queries)
    t = data.load_sift(tmp_path / "t.hdf5", limit=200, n_test=50, n_queries=10)
    assert t.truth[0, 0] == 7


def test_the_split_needs_enough_rows():
    with pytest.raises(ValueError):
        data.make_split(make_x(n=100), n_test=100, n_queries=10)

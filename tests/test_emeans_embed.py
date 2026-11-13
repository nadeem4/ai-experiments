"""Embedding helpers, shards and the loader, with a stub model and no downloads."""
import json

import numpy as np
import pytest

from emeans import data, embed


class Stub:
    def __init__(self, d=8):
        self.d, self.seen, self.calls = d, [], 0

    def encode(self, texts, batch_size=32, normalize_embeddings=True, **kw):
        self.calls += 1
        self.seen += list(texts)
        rng = np.random.default_rng(len(self.seen))
        return rng.normal(size=(len(texts), self.d)).astype(np.float32) * 3  # deliberately not unit


def sample():
    return [{"id": f"a{i}", "month": m, "abstract": f"text {i}"}
            for i, m in enumerate(["2018-01", "2018-01", "2018-02"])]


def test_prefix_on_every_input_and_unit_norm():
    stub = Stub()
    x = embed.embed_month(stub, ["a", "b"], batch_size=2)
    assert all(t.startswith("search_document: ") for t in stub.seen)
    assert x.dtype == np.float32
    assert np.allclose(np.linalg.norm(x, axis=1), 1, atol=1e-5)


def test_run_writes_outputs_and_is_resumable(tmp_path):
    stub = Stub()
    embed.run(stub, sample(), tmp_path, revision="abc")
    assert (tmp_path / "shards" / "2018-01.npz").exists()
    assert stub.calls == 2
    again = Stub()
    embed.run(again, sample(), tmp_path, revision="abc")
    assert again.calls == 0
    x, months = data.load_arxiv(tmp_path / "vectors.npy")
    assert x.shape == (3, 8) and list(months) == ["2018-01", "2018-01", "2018-02"]
    assert (tmp_path / "ids.txt").read_text().split() == ["a0", "a1", "a2"]
    assert json.loads((tmp_path / "meta.json").read_text())["revision"] == "abc"


def test_truncate_shapes_and_norm():
    x = np.random.default_rng(0).normal(size=(5, 768)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    t = data.truncate(x, 64)
    assert t.shape == (5, 64) and t.dtype == np.float32
    assert np.allclose(np.linalg.norm(t, axis=1), 1, atol=1e-5)
    assert np.array_equal(data.truncate(x, 768), x)


def test_month_order_is_stable():
    months = np.array(["2018-02", "2018-01", "2018-02", "2018-01"])
    assert list(data.month_order(months)) == [1, 3, 0, 2]

"""Embedding helpers for the Kaggle notebook: per-month shards, then the final arrays.

Imports no faiss, h5py or sentence-transformers: the model is passed in, so the
tests use a stub and the kernel passes the real one.
"""
import json
from pathlib import Path

import numpy as np

PREFIX = "search_document: "
MODEL = "nomic-ai/nomic-embed-text-v1.5"
# "main" is resolved to a commit hash by the kernel at run time and recorded in
# meta.json; replace this with that hash to pin it in the repository.
REVISION = "main"


def shard_path(out_dir, month):
    return Path(out_dir) / "shards" / f"{month}.npz"


def embed_month(model, texts, batch_size=64):
    """float32 (n, d), unit norm exactly (re-normalised in float32 after encoding)."""
    x = np.asarray(model.encode([PREFIX + t for t in texts], batch_size=batch_size,
                                normalize_embeddings=True), dtype=np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def run(model, sample, out_dir, batch_size=64, revision=REVISION):
    """Embed `sample` (sorted by month) one month at a time; a month whose shard
    exists is skipped, so an interrupted run resumes. Writes vectors.npy,
    months.npy, ids.txt and meta.json."""
    out = Path(out_dir)
    by_month = {}
    for r in sample:
        by_month.setdefault(r["month"], []).append(r)
    vectors, months, ids = [], [], []
    for month, rows in sorted(by_month.items()):
        path = shard_path(out, month)
        if path.exists():
            x = np.load(path)["x"]
        else:
            x = embed_month(model, [r["abstract"] for r in rows], batch_size)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, x=x)
            tmp.replace(path)
        vectors.append(x)
        months += [month] * len(rows)
        ids += [r["id"] for r in rows]
    np.save(out / "vectors.npy", np.concatenate(vectors))
    np.save(out / "months.npy", np.array(months))
    (out / "ids.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    (out / "meta.json").write_text(json.dumps({"model": MODEL, "revision": revision,
                                               "dimensions": int(vectors[0].shape[1])}),
                                   encoding="utf-8")

"""Rerankers. Every reranker has a `name` and one method:

    score(query, passage) -> {"score": float, "request": {...}, "response": {...},
                              "latency_ms": float}

`passage` is {"title": ..., "text": ...}. A higher score means more relevant;
the scale is the reranker's own and is never compared across rerankers, only
used to sort one query's candidates.

`request` and `response` are the exact wire: what was sent and what came back,
recorded verbatim for every pair scored. Model-specific code lives in its own
module, and this file is the only place a reranker is chosen.

A reranker that calls a remote model may also return `retries` and `failed`. A
failed call carries `score: None` -- nothing is invented for it, and the ranking
leaves that passage where BM25 put it.
"""
from .jev import VARIANTS as JEV_VARIANTS
from .laya import VARIANTS as LAYA_VARIANTS

RERANKER_NAMES = list(LAYA_VARIANTS) + list(JEV_VARIANTS) + ["cross-encoder"]


def make_reranker(name, **kwargs):
    if name in LAYA_VARIANTS:
        from .laya import load
        return load(name, **kwargs)
    if name in JEV_VARIANTS:
        from .jev import load
        return load(name, **kwargs)
    if name == "cross-encoder":
        from .cross_encoder import load
        return load(**kwargs)
    raise ValueError(f"unknown reranker: {name}")

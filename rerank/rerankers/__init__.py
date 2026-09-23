"""Rerankers. Every reranker has a `name` and one method:

    score(query, passage) -> {"score": float, "request": {...}, "response": {...},
                              "latency_ms": float}

`passage` is {"title": ..., "text": ...}. A higher score means more relevant;
the scale is the reranker's own and is never compared across rerankers, only
used to sort one query's candidates.

`request` and `response` are the exact wire: what was sent and what came back,
recorded verbatim for every pair scored. Model-specific code lives in its own
module, and this file is the only place a reranker is chosen -- adding Jev means
a new module and one line here, with no change to the task.
"""
RERANKER_NAMES = ["laya-score", "laya-noul", "cross-encoder"]


def make_reranker(name, **kwargs):
    if name in ("laya-score", "laya-noul"):
        from .laya import load
        return load(name, **kwargs)
    if name == "cross-encoder":
        from .cross_encoder import load
        return load(**kwargs)
    raise ValueError(f"unknown reranker: {name}")

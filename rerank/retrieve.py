"""The first stage: what the re-rankers are given.

BM25 alone is not how retrieval is served. A production pipeline runs a lexical
and a dense retriever and fuses them, and on this corpus that is worth 20 queries
-- the ones where BM25's top 20 held nothing relevant at all, so no re-ranker
could move them and they diluted every mean.

This is not what the experiment measures. It is held fixed so that the scorer is
the only thing that varies, and its settings are written into the run config so
the candidate set can be rebuilt.
"""
ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60
DEPTH = 100  # how deep each half goes before fusion, not how many survive it
PASSAGE_CHARS = 1000  # the same cut the scorers see


def settings():
    return {"first_stage": "hybrid", "encoder": ENCODER, "fusion": "reciprocal-rank",
            "rrf_k": RRF_K, "depth": DEPTH}


def fuse(rankings, k=RRF_K):
    """Reciprocal rank fusion over {half: [doc_id best first]}.

    Rank-based rather than score-based because a BM25 score and a cosine
    similarity are not on the same scale, and normalising them into one is a
    tuning knob that would have to be justified and would not be."""
    scores = {}
    for ranking in rankings.values():
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # doc_id breaks ties, so the same inputs always give the same order.
    return sorted(scores, key=lambda d: (-scores[d], d))


def _encoder(device=None):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(ENCODER, device=device)


def dense_index(corpus, device=None, log=lambda *_: None):
    """-> (doc_ids, embeddings), normalised so a dot product is cosine."""
    import numpy as np

    model = _encoder(device)
    doc_ids = list(corpus)
    texts = [f"{corpus[d].get('title', '')} {corpus[d].get('text', '')}"[:PASSAGE_CHARS]
             for d in doc_ids]
    log(f"embedding {len(texts)} documents with {ENCODER}")
    vectors = model.encode(texts, batch_size=64, normalize_embeddings=True,
                           show_progress_bar=False)
    return doc_ids, np.asarray(vectors), model


def build_candidates(corpus, queries, top_k, log=lambda *_: None, device=None):
    """-> ({query_id: [doc_id]}, seconds, {query_id: {doc_id: bm25 score}}).

    The BM25 scores come back too: re-ranking the fused candidates by lexical
    score alone is then a baseline that costs nothing, since they are already
    computed."""
    import time

    import numpy as np

    from .bm25 import BM25Index, tokenize

    started = time.perf_counter()
    lexical = BM25Index(corpus)
    doc_ids, vectors, model = dense_index(corpus, device, log)
    index_of = {d: i for i, d in enumerate(doc_ids)}

    query_ids = list(queries)
    query_vectors = model.encode([queries[q] for q in query_ids],
                                 normalize_embeddings=True, show_progress_bar=False)
    similarity = np.asarray(query_vectors) @ vectors.T

    candidates, bm25_scores = {}, {}
    for position, query_id in enumerate(query_ids):
        raw = lexical.bm25.get_scores(tokenize(queries[query_id]))
        order = sorted(range(len(lexical.doc_ids)), key=lambda i: (-raw[i], i))[:DEPTH]
        lexical_top = [lexical.doc_ids[i] for i in order]
        dense_top = [doc_ids[i] for i in np.argsort(-similarity[position])[:DEPTH]]

        fused = fuse({"lexical": lexical_top, "dense": dense_top})[:top_k]
        candidates[query_id] = fused
        bm25_scores[query_id] = {d: float(raw[index_of[d]]) for d in fused}

    seconds = time.perf_counter() - started
    log(f"hybrid top-{top_k} for {len(queries)} queries in {seconds:.1f}s")
    return candidates, seconds, bm25_scores

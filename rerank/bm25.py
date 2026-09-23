"""BM25 candidate generation with rank_bm25.

NFCorpus is 3,633 documents. Pyserini would be the stronger index but it drags in
a JVM; at this size the pure-Python one is a few seconds and the candidate set is
what matters, not how fast it was built.

The order this returns is the BM25 ranking -- the floor every other method is
measured against, and the tie-break every other method falls back to.
"""
import re

TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN.findall(text.lower())


class BM25Index:
    def __init__(self, corpus):
        from rank_bm25 import BM25Okapi
        self.doc_ids = list(corpus)
        self.bm25 = BM25Okapi([tokenize(f"{d.get('title', '')} {d.get('text', '')}") for d in corpus.values()])

    def top_k(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.doc_ids)), key=lambda i: (-scores[i], i))[:k]
        return [self.doc_ids[i] for i in order]

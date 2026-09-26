"""Candidate generation. Every ranking re-ranks this same list, so it decides the
ceiling for all of them."""
from rerank.bm25 import BM25Index, tokenize

CORPUS = {
    "d1": {"title": "Vitamin D", "text": "vitamin d supplements and bone health"},
    "d2": {"title": "Cancer risk", "text": "smoking raises lung cancer risk"},
    "d3": {"title": "Diet", "text": "a plant based diet and cholesterol"},
}


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Vitamin-D, really?") == ["vitamin", "d", "really"]


def test_the_matching_document_comes_first():
    assert BM25Index(CORPUS).top_k("lung cancer smoking", 3)[0] == "d2"


def test_top_k_limits_the_candidates():
    assert len(BM25Index(CORPUS).top_k("cancer", 2)) == 2


def test_asking_for_more_than_the_corpus_returns_the_corpus():
    assert len(BM25Index(CORPUS).top_k("cancer", 99)) == 3


def test_candidate_order_is_deterministic():
    index = BM25Index(CORPUS)
    assert index.top_k("diet cholesterol", 3) == index.top_k("diet cholesterol", 3)

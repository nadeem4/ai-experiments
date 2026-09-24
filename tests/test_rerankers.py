"""Prompt construction and the score a reranker reads back out of an answer.

No model is loaded here: a reranker is handed a fake `predict` so the request
that goes on the wire, and the number that comes back, are both checkable.
"""
import pytest

from rerank.rerankers import make_reranker
from rerank.rerankers.laya import SCORE_QUESTION, LayaReranker, build_state

PASSAGE = {"title": "Vitamin D and cancer", "text": "A review of vitamin D intake." * 200}


def test_state_carries_query_and_passage():
    state = build_state("does vitamin d prevent cancer", PASSAGE)
    assert state["query"] == "does vitamin d prevent cancer"
    assert state["passage"].startswith("Vitamin D and cancer")


def test_passage_is_truncated_to_fit_the_512_token_context():
    """The English checkpoint reads 512 tokens, ~320 of them state. An untruncated
    NFCorpus abstract silently loses its tail; we cut it where we can see it."""
    state = build_state("q", PASSAGE, max_chars=300)
    assert len(state["passage"]) <= 300


def test_an_empty_title_is_not_glued_on():
    state = build_state("q", {"title": "", "text": "body"})
    assert state["passage"] == "body"


def test_score_question_rungs_run_from_irrelevant_to_answers_it():
    criteria = SCORE_QUESTION["relevance"]["criteria"]
    assert isinstance(criteria, list) and len(criteria) >= 3
    assert "irrelevant" in criteria[0].lower()


def test_score_formulation_reads_the_score_field():
    fake = lambda state, questions: {"answers": {"relevance": {"type": "score", "score": 2.25}}}
    r = LayaReranker("laya-score", SCORE_QUESTION, "score", predict=fake)
    out = r.score("q", PASSAGE)
    assert out["score"] == 2.25


def test_noul_formulation_reads_the_probability():
    fake = lambda state, questions: {"answers": {"relevance": {"type": "noul", "noul": 0.83}}}
    r = make_reranker("laya-noul", predict=fake)
    assert r.score("q", PASSAGE)["score"] == 0.83


def test_the_exact_request_and_response_are_returned_for_the_record():
    seen = {}

    def fake(state, questions):
        seen["state"], seen["questions"] = state, questions
        return {"answers": {"relevance": {"type": "score", "score": 1.0}}, "usage": {"input_tokens": 7}}

    out = make_reranker("laya-score", predict=fake).score("cancer", PASSAGE)
    assert out["request"] == {"state": seen["state"], "questions": seen["questions"]}
    assert out["response"]["usage"] == {"input_tokens": 7}
    assert out["latency_ms"] > 0


def test_unknown_reranker_is_an_error():
    with pytest.raises(ValueError):
        make_reranker("laya-typed")


def test_the_typed_decisions_variants_ask_the_same_two_questions_as_the_base():
    """Same questions, same passage cut, a different checkpoint -- so the two
    checkpoints can be compared on the identical candidates."""
    fake = lambda state, questions: {"answers": {"relevance": {"type": "score", "score": 3.0}}}
    base, typed = make_reranker("laya-score", predict=fake), make_reranker("laya-typed-score", predict=fake)
    assert typed.questions == base.questions
    assert typed.field == base.field
    assert typed.score("q", PASSAGE)["request"] == base.score("q", PASSAGE)["request"]

    noul = make_reranker("laya-typed-noul", predict=lambda s, q: {"answers": {"relevance": {"noul": 0.2}}})
    assert noul.field == "noul"
    assert noul.score("q", PASSAGE)["score"] == 0.2


def test_the_typed_decisions_variants_load_the_typed_decisions_subfolder():
    from rerank.rerankers.laya import VARIANTS
    assert VARIANTS["laya-typed-score"][2] == "typed-decisions"
    assert VARIANTS["laya-typed-noul"][2] == "typed-decisions"
    assert VARIANTS["laya-score"][2] is None

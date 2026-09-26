"""Jev as a reranker: the wire, the retry, and what a failed call leaves behind.

No network here. The HTTP post is a fake, so the exact body sent, the retry
behaviour under a rate limit, and the latency that is reported are all checkable.
"""
import urllib.error

import pytest

from rerank.rerankers import make_reranker
from rerank.rerankers.jev import (OPENROUTER_MODEL, OPENROUTER_URL, TYPESAFE_MODEL,
                                  TYPESAFE_URL, JevReranker)

PASSAGE = {"title": "Vitamin D and cancer", "text": "A review of vitamin D intake."}
ANSWER_SCORE = {"answers": {"relevance": {"type": "score", "score": 2.25}},
                "usage": {"input_tokens": 233, "output_tokens": 0}}


def _http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, None)


def test_posts_the_same_state_and_question_laya_gets():
    from rerank.rerankers.laya import SCORE_QUESTION, build_state
    sent = {}

    def post(url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        return ANSWER_SCORE

    out = make_reranker("jev-score", api_key="k", post=post).score("cancer?", PASSAGE)
    assert sent["url"] == OPENROUTER_URL
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["body"] == {"model": OPENROUTER_MODEL,
                            "state": build_state("cancer?", PASSAGE),
                            "questions": SCORE_QUESTION}
    assert out["score"] == 2.25


def test_the_openrouter_build_is_pinned_to_a_date():
    """`typesafe/jev-1.13` is a floating minor and `~typesafe/jev-latest` is an
    alias. A benchmark that cannot say which build produced a number is not a
    benchmark, so the dated build is what goes on the wire."""
    assert OPENROUTER_MODEL.startswith("typesafe/jev-")
    assert OPENROUTER_MODEL[-8:].isdigit(), f"{OPENROUTER_MODEL} carries no dated build"


def test_typesafe_route_pins_the_model_version():
    sent = {}

    def post(url, headers, body):
        sent.update(url=url, body=body)
        return ANSWER_SCORE

    make_reranker("jev-score", api_key="ts", provider="typesafe", post=post).score("q", PASSAGE)
    assert sent["url"] == TYPESAFE_URL
    assert sent["body"]["model"] == TYPESAFE_MODEL


def test_noul_is_sent_as_jevs_boolean_and_read_back_as_a_probability():
    """Laya calls a no-criteria yes/no question `noul` and answers with `noul`;
    Jev's v4 schema calls the identical question `boolean` and answers with
    `probability`. A `noul` body comes back 400. Same question, two dialects --
    so the instructions must still be word for word Laya's."""
    from rerank.rerankers.laya import NOUL_QUESTION
    sent = {}

    def post(url, headers, body):
        sent.update(body=body)
        return {"answers": {"relevance": {"type": "boolean", "probability": 0.83}}}

    out = make_reranker("jev-noul", api_key="k", post=post).score("q", PASSAGE)
    assert out["score"] == 0.83
    assert sent["body"]["questions"]["relevance"]["type"] == "boolean"
    assert (sent["body"]["questions"]["relevance"]["instructions"]
            == NOUL_QUESTION["relevance"]["instructions"])


def test_the_exact_request_and_response_are_recorded():
    out = make_reranker("jev-score", api_key="k", post=lambda *a: ANSWER_SCORE).score("q", PASSAGE)
    assert out["response"] == ANSWER_SCORE
    assert out["request"]["state"]["query"] == "q"
    assert out["latency_ms"] > 0


def test_rate_limits_are_retried_and_counted():
    calls = []

    def post(url, headers, body):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429 if len(calls) == 1 else 529)
        return ANSWER_SCORE

    out = JevReranker("jev-score", {}, "score", api_key="k", post=post, backoff_s=0).score("q", PASSAGE)
    assert len(calls) == 3
    assert out["retries"] == 2
    assert out["failed"] is False


def test_latency_excludes_the_time_spent_waiting_out_a_rate_limit():
    """The reported number has to be the model's, not the queue's."""
    import time
    calls = []

    def post(url, headers, body):
        calls.append(1)
        if len(calls) == 1:
            raise _http_error(429)
        return ANSWER_SCORE

    r = JevReranker("jev-score", {}, "score", api_key="k", post=post, backoff_s=0.2)
    started = time.perf_counter()
    out = r.score("q", PASSAGE)
    assert (time.perf_counter() - started) * 1000 > 190  # the wait really happened
    assert out["latency_ms"] < 100  # and it is not in the reported latency


def test_a_call_that_never_succeeds_is_recorded_as_failed_with_no_score():
    def post(url, headers, body):
        raise _http_error(429)

    out = JevReranker("jev-score", {}, "score", api_key="k", post=post,
                      backoff_s=0, max_retries=2).score("q", PASSAGE)
    assert out["failed"] is True
    assert out["score"] is None
    assert out["retries"] == 2
    assert "429" in json_text(out["response"])


def test_an_error_that_is_not_a_rate_limit_is_not_retried():
    calls = []

    def post(url, headers, body):
        calls.append(1)
        raise _http_error(401)

    out = JevReranker("jev-score", {}, "score", api_key="k", post=post, backoff_s=0).score("q", PASSAGE)
    assert len(calls) == 1
    assert out["failed"] is True


def test_a_reranker_with_no_key_is_an_error():
    """Better a name-the-variable error up front than 1,500 401s."""
    with pytest.raises(ValueError):
        JevReranker("jev-score", {}, "score", api_key=None, post=lambda *a: ANSWER_SCORE)


def json_text(response):
    import json
    return json.dumps(response)


def test_the_env_file_is_looked_for_at_this_repos_root():
    """It used to be the parent repo's root. A path that walks out of this repo
    would silently read nothing, so the depth is pinned."""
    from pathlib import Path
    from rerank.rerankers import jev
    assert jev.ROOT_ENV.name == ".env"
    assert (jev.ROOT_ENV.parent / "pyproject.toml").exists(), f"{jev.ROOT_ENV} is not this repo's root"

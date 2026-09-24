"""The wire: what is sent, what is retried, and which clock each number comes off.

No network here. The HTTP post is a fake, so the exact body, the retry behaviour
under a rate limit, and the two separately reported clocks are all checkable.
Reported latency covers only the attempt that succeeded; wall clock covers the
whole call including backoff, so waiting out a queue is never charged to a model's
speed and never quietly disappears either.
"""
import urllib.error

import pytest

from latency import gateway

CRITERIA = {"IDLE": "Keep your lane and speed.", "FASTER": "Accelerate."}
CHAT_OK = {
    "choices": [{"message": {"content": '{"choice": "IDLE"}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 110, "completion_tokens": 7, "market_cost": 1.38e-05},
}
JEV_OK = {
    "answers": {gateway.QUESTION_ID: {"type": "choice", "choice": "IDLE",
                                      "probabilities": {"IDLE": 0.5, "FASTER": 0.5}}},
    "usage": {"inputTokens": 492, "outputTokens": 61},
    "providerMetadata": {"gateway": {"marketCost": "0.000020664"}},
}


def _http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, None)


def test_a_chat_call_asks_for_structured_output_at_temperature_zero():
    sent = {}

    def post(url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        return CHAT_OK

    model = gateway.ChatModel("gpt", "openai/gpt-4.1-nano", api_key="k", post=post)
    out = model.decide("the state", "Pick one.", CRITERIA)
    assert sent["url"] == gateway.CHAT_URL
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["body"]["model"] == "openai/gpt-4.1-nano"
    assert sent["body"]["temperature"] == 0
    assert sent["body"]["response_format"]["json_schema"]["schema"]["properties"]["choice"]["enum"] == ["IDLE", "FASTER"]
    assert out["content"] == '{"choice": "IDLE"}'
    assert out["structured"] is True


def test_a_model_that_refuses_temperature_is_asked_without_it_and_recorded_as_such():
    """`gpt-5` and the newer Claudes reject a temperature. Sending one anyway would
    fail the call; pretending they ran at zero would be a lie in the config."""
    sent = {}

    def post(url, headers, body):
        sent.update(body=body)
        return CHAT_OK

    model = gateway.ChatModel("gpt", "openai/gpt-5", api_key="k", post=post, temperature=None)
    out = model.decide("s", "i", CRITERIA)
    assert "temperature" not in sent["body"]
    assert out["temperature"] is None


def test_a_chat_call_reports_the_providers_own_tokens_and_the_gateways_own_cost():
    model = gateway.ChatModel("gpt", "m", api_key="k", post=lambda *a: CHAT_OK)
    out = model.decide("s", "i", CRITERIA)
    assert (out["input_tokens"], out["output_tokens"]) == (110, 7)
    assert out["market_cost"] == pytest.approx(1.38e-05)


def test_a_provider_that_reports_no_tokens_leaves_them_none_rather_than_estimated():
    model = gateway.ChatModel("gpt", "m", api_key="k", post=lambda *a: {"choices": [{"message": {"content": "{}"}}]})
    out = model.decide("s", "i", CRITERIA)
    assert out["input_tokens"] is None and out["output_tokens"] is None and out["market_cost"] is None


def test_a_rate_limit_is_retried_and_the_backoff_is_not_charged_to_latency():
    calls, slept = [], []

    def post(url, headers, body):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429)
        return CHAT_OK

    model = gateway.ChatModel("gpt", "m", api_key="k", post=post, backoff_s=10.0, sleep=slept.append)
    out = model.decide("s", "i", CRITERIA)
    assert out["retries"] == 2 and out["failed"] is False
    assert slept == [10.0, 20.0]  # deliberately outside the timer
    assert out["latency_ms"] < 1000  # the successful attempt only
    assert out["wall_ms"] >= out["latency_ms"]


def test_an_error_that_is_not_retryable_stops_at_once():
    calls = []

    def post(url, headers, body):
        calls.append(1)
        raise _http_error(403)

    model = gateway.ChatModel("gpt", "m", api_key="k", post=post, sleep=lambda s: None)
    out = model.decide("s", "i", CRITERIA)
    assert len(calls) == 1
    assert out["failed"] is True and out["content"] is None
    assert "403" in out["error"]


def test_a_call_that_never_succeeds_is_recorded_as_failed_not_raised():
    def post(url, headers, body):
        raise _http_error(429)

    model = gateway.ChatModel("gpt", "m", api_key="k", post=post, max_retries=2, sleep=lambda s: None)
    out = model.decide("s", "i", CRITERIA)
    assert out["failed"] is True and out["retries"] == 2 and out["latency_ms"] is None


def test_the_answer_is_never_asked_for_twice():
    """One call, one answer. A repaired answer is a different product with a
    different latency, so an invalid completion is a result, not a reason to retry."""
    calls = []

    def post(url, headers, body):
        calls.append(1)
        return {"choices": [{"message": {"content": "I would keep my lane."}}], "usage": {}}

    gateway.ChatModel("gpt", "m", api_key="k", post=post).decide("s", "i", CRITERIA)
    assert len(calls) == 1


def test_jev_is_sent_the_typed_question_itself_not_a_rendering_of_it():
    sent = {}

    def post(url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        return JEV_OK

    out = gateway.JevModel("jev", "typesafe-ai/jev", api_key="k", post=post).decide("the state", "Pick one.", CRITERIA)
    assert sent["url"] == gateway.EVALUATION_URL
    assert sent["headers"]["ai-model-id"] == "typesafe-ai/jev"
    assert sent["headers"]["ai-evaluation-model-specification-version"] == "4"
    assert sent["body"] == {
        "state": "the state",
        "questions": {gateway.QUESTION_ID: {"type": "choice", "instructions": "Pick one.",
                                            "criteria": CRITERIA}},
    }
    assert out["choice"] == "IDLE"
    assert out["structured"] == "typed"


def test_jev_reports_its_own_tokens_and_the_gateways_market_cost():
    out = gateway.JevModel("jev", "typesafe-ai/jev", api_key="k", post=lambda *a: JEV_OK).decide("s", "i", CRITERIA)
    assert (out["input_tokens"], out["output_tokens"]) == (492, 61)
    assert out["market_cost"] == pytest.approx(2.0664e-05)


def test_jev_cannot_answer_off_the_option_list():
    """It scores the options it was given, so `choice` is an option by construction.
    That is the property the experiment is measuring, so it is asserted, not hoped."""
    out = gateway.JevModel("jev", "typesafe-ai/jev", api_key="k", post=lambda *a: JEV_OK).decide("s", "i", CRITERIA)
    assert out["choice"] in CRITERIA

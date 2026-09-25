"""The wire: what is sent, what is retried, and which clock each number comes off.

No network here. The HTTP post is a fake, so the exact body, the retry behaviour
under a rate limit, and the two separately reported clocks are all checkable.
Reported latency covers only the attempt that succeeded; wall clock covers the
whole call including backoff, so waiting out a queue is never charged to a model's
speed and never quietly disappears either.

The two routes are deliberately different endpoints: the LLMs go to
`chat/completions`, Jev goes to `systemone`, which is the only route that accepts
the decision shape at all.
"""
import urllib.error

import pytest

from decision_cost import openrouter

CRITERIA = {"IDLE": "Keep your lane and speed.", "FASTER": "Accelerate."}
CHAT_OK = {
    "choices": [{"message": {"content": '{"choice": "IDLE"}'}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 110, "completion_tokens": 7, "cost": 1.38e-05,
              "completion_tokens_details": {"reasoning_tokens": 0}},
}
JEV_OK = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {openrouter.QUESTION_ID: {"type": "choice", "choice": "IDLE",
                                         "probabilities": {"IDLE": 0.5, "FASTER": 0.5},
                                         "confidence": 0.82}},
    "usage": {"input_tokens": 400, "output_tokens": 62, "cost": 1.68e-05},
}


def _http_error(code):
    return urllib.error.HTTPError("u", code, "err", {}, None)


# --- the LLM route ----------------------------------------------------------


def test_a_chat_call_goes_to_chat_completions_with_a_bearer_token():
    sent = {}

    def post(url, headers, body):
        sent.update(url=url, headers=headers, body=body)
        return CHAT_OK

    model = openrouter.ChatModel("gpt", "openai/gpt-6-luna", api_key="k", post=post)
    out = model.decide("the state", "Pick one.", CRITERIA)
    assert sent["url"] == openrouter.CHAT_URL == "https://openrouter.ai/api/v1/chat/completions"
    assert sent["headers"]["Authorization"] == "Bearer k"
    assert sent["body"]["model"] == "openai/gpt-6-luna"
    assert out["content"] == '{"choice": "IDLE"}'
    assert out["structured"] is True


def test_a_chat_call_keeps_the_strict_enum_of_option_keys():
    """Fairness rule 1, unchanged by the migration: the LLM is constrained to the
    option list, which is the strongest constraint the transport exposes."""
    sent = {}

    def post(url, headers, body):
        sent.update(body=body)
        return CHAT_OK

    openrouter.ChatModel("gpt", "m", api_key="k", post=post).decide("s", "i", CRITERIA)
    schema = sent["body"]["response_format"]["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["properties"]["choice"]["enum"] == ["IDLE", "FASTER"]


def test_a_model_that_accepts_a_temperature_is_asked_at_zero():
    sent = {}

    def post(url, headers, body):
        sent.update(body=body)
        return CHAT_OK

    model = openrouter.ChatModel("gemma", "google/gemma-4-31b-it", api_key="k", post=post,
                                 temperature=0)
    assert model.decide("s", "i", CRITERIA)["temperature"] == 0
    assert sent["body"]["temperature"] == 0


def test_a_model_that_refuses_temperature_is_asked_without_it_and_recorded_as_such():
    """The current GPT and Claude tiers reject a temperature. Sending one anyway
    would fail the call; pretending they ran at zero would be a lie in the config."""
    sent = {}

    def post(url, headers, body):
        sent.update(body=body)
        return CHAT_OK

    model = openrouter.ChatModel("gpt", "openai/gpt-6-luna", api_key="k", post=post,
                                 temperature=None)
    out = model.decide("s", "i", CRITERIA)
    assert "temperature" not in sent["body"]
    assert out["temperature"] is None


def test_reasoning_effort_is_sent_only_to_a_model_that_advertises_it():
    """The current tiers are reasoning models and their thinking tokens can eat the
    answer budget whole -- one model returned no answer at all inside 64 tokens.
    Asking for a decision rather than an essay is the same shape as the temperature
    rule: sent where advertised, recorded either way."""
    sent = {}

    def post(url, headers, body):
        sent.update(body=body)
        return CHAT_OK

    on = openrouter.ChatModel("gpt", "m", api_key="k", post=post, reasoning_effort="minimal")
    assert on.decide("s", "i", CRITERIA)["reasoning_effort"] == "minimal"
    assert sent["body"]["reasoning_effort"] == "minimal"

    sent.clear()
    off = openrouter.ChatModel("gemma", "m", api_key="k", post=post, reasoning_effort=None)
    assert off.decide("s", "i", CRITERIA)["reasoning_effort"] is None
    assert "reasoning_effort" not in sent["body"]


def test_a_chat_call_reports_openrouters_own_tokens_and_per_call_cost():
    """Cost is never computed from a token count and a rate off a pricing page."""
    model = openrouter.ChatModel("gpt", "m", api_key="k", post=lambda *a: CHAT_OK)
    out = model.decide("s", "i", CRITERIA)
    assert (out["input_tokens"], out["output_tokens"]) == (110, 7)
    assert out["cost"] == pytest.approx(1.38e-05)


def test_reasoning_tokens_are_recorded_so_a_truncated_answer_is_explicable():
    payload = {**CHAT_OK, "usage": {**CHAT_OK["usage"],
                                    "completion_tokens_details": {"reasoning_tokens": 57}}}
    out = openrouter.ChatModel("g", "m", api_key="k",
                               post=lambda *a: payload).decide("s", "i", CRITERIA)
    assert out["reasoning_tokens"] == 57


def test_a_provider_that_reports_no_tokens_leaves_them_none_rather_than_estimated():
    model = openrouter.ChatModel("gpt", "m", api_key="k",
                                 post=lambda *a: {"choices": [{"message": {"content": "{}"}}]})
    out = model.decide("s", "i", CRITERIA)
    assert out["input_tokens"] is None and out["output_tokens"] is None and out["cost"] is None


# --- the two clocks and the retry loop ---------------------------------------


def test_a_rate_limit_is_retried_and_the_backoff_is_not_charged_to_latency():
    calls, slept = [], []

    def post(url, headers, body):
        calls.append(1)
        if len(calls) < 3:
            raise _http_error(429)
        return CHAT_OK

    model = openrouter.ChatModel("gpt", "m", api_key="k", post=post, backoff_s=10.0,
                                 sleep=slept.append)
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

    model = openrouter.ChatModel("gpt", "m", api_key="k", post=post, sleep=lambda s: None)
    out = model.decide("s", "i", CRITERIA)
    assert len(calls) == 1
    assert out["failed"] is True and out["content"] is None
    assert "403" in out["error"]


def test_a_call_that_never_succeeds_is_recorded_as_failed_not_raised():
    def post(url, headers, body):
        raise _http_error(429)

    model = openrouter.ChatModel("gpt", "m", api_key="k", post=post, max_retries=2,
                                 sleep=lambda s: None)
    out = model.decide("s", "i", CRITERIA)
    assert out["failed"] is True and out["retries"] == 2 and out["latency_ms"] is None


def test_the_answer_is_never_asked_for_twice():
    """One call, one answer. A repaired answer is a different product with a
    different latency, so an invalid completion is a result, not a reason to retry."""
    calls = []

    def post(url, headers, body):
        calls.append(1)
        return {"choices": [{"message": {"content": "I would keep my lane."}}], "usage": {}}

    openrouter.ChatModel("gpt", "m", api_key="k", post=post).decide("s", "i", CRITERIA)
    assert len(calls) == 1


# --- the decision route ------------------------------------------------------


def test_jev_goes_to_systemone_not_to_chat_completions():
    """`chat/completions` rejects the decision shape outright, so the route is the
    whole difference between a working call and a 400."""
    sent = {}

    def post(url, headers, body):
        sent.update(url=url, body=body)
        return JEV_OK

    openrouter.JevModel("jev", openrouter.JEV_ID, api_key="k", post=post).decide(
        "the state", "Pick one.", CRITERIA)
    assert sent["url"] == openrouter.SYSTEMONE_URL == "https://openrouter.ai/api/v1/systemone"
    assert sent["url"] != openrouter.CHAT_URL


def test_jev_is_sent_the_typed_question_itself_not_a_rendering_of_it():
    sent = {}

    def post(url, headers, body):
        sent.update(body=body)
        return JEV_OK

    out = openrouter.JevModel("jev", "typesafe/jev-1.13-20260917", api_key="k",
                              post=post).decide("the state", "Pick one.", CRITERIA)
    assert sent["body"] == {
        "model": "typesafe/jev-1.13-20260917",
        "state": "the state",
        "questions": {openrouter.QUESTION_ID: {"type": "choice", "instructions": "Pick one.",
                                               "criteria": CRITERIA}},
    }
    assert out["choice"] == "IDLE"
    assert out["structured"] == "typed"


def test_jev_carries_the_model_id_in_the_body_not_in_a_header():
    """The Vercel route took `ai-model-id` as a header. OpenRouter takes `model` in
    the body, like every other route it serves."""
    sent = {}

    def post(url, headers, body):
        sent.update(headers=headers, body=body)
        return JEV_OK

    openrouter.JevModel("jev", openrouter.JEV_ID, api_key="k",
                        post=post).decide("s", "i", CRITERIA)
    assert sent["body"]["model"] == openrouter.JEV_ID
    assert "ai-model-id" not in sent["headers"]


def test_jev_reports_its_own_tokens_and_openrouters_per_call_cost():
    """`usage.cost` in dollars, taken as given -- never recomputed from a price."""
    out = openrouter.JevModel("jev", "m", api_key="k",
                              post=lambda *a: JEV_OK).decide("s", "i", CRITERIA)
    assert (out["input_tokens"], out["output_tokens"]) == (400, 62)
    assert out["cost"] == pytest.approx(1.68e-05)


def test_jev_cannot_answer_off_the_option_list():
    """It scores the options it was given, so `choice` is an option by construction.
    That is the property the experiment is measuring, so it is asserted, not hoped."""
    out = openrouter.JevModel("jev", "m", api_key="k",
                              post=lambda *a: JEV_OK).decide("s", "i", CRITERIA)
    assert out["choice"] in CRITERIA


def test_jev_keeps_the_probability_distribution_not_just_the_argmax():
    out = openrouter.JevModel("jev", "m", api_key="k",
                              post=lambda *a: JEV_OK).decide("s", "i", CRITERIA)
    assert out["probabilities"] == {"IDLE": 0.5, "FASTER": 0.5}


# --- the transport tag -------------------------------------------------------


def test_a_chat_call_records_the_transport_it_went_over():
    """The transport changed, so these numbers are not comparable to the Vercel
    pilot's. Tagging every record is what makes averaging the two impossible."""
    out = openrouter.ChatModel("gpt", "m", api_key="k",
                               post=lambda *a: CHAT_OK).decide("s", "i", CRITERIA)
    assert out["transport"] == openrouter.TRANSPORT == "openrouter"


def test_a_jev_call_records_the_transport_it_went_over():
    out = openrouter.JevModel("jev", "m", api_key="k",
                              post=lambda *a: JEV_OK).decide("s", "i", CRITERIA)
    assert out["transport"] == "openrouter"


def test_a_missing_key_is_refused_up_front_rather_than_at_the_first_call():
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        openrouter.ChatModel("gpt", "m", api_key=None)

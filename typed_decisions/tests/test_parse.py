"""Did the call return one of the allowed options, without post-hoc repair?

This is a first-class result, not an error to hide. So the rules are strict on
purpose: the answer is the `choice` field of a JSON object, spelled exactly as the
option is spelled. Nothing is fuzzy-matched, nothing is retried, and a model that
answers with prose or with a key of its own invention has not done the job.
"""
from decision_cost import parse

OPTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def test_a_well_formed_answer_is_valid():
    assert parse.classify('{"choice": "IDLE"}', OPTIONS) == {"choice": "IDLE", "validity": "valid", "detail": None}


def test_surrounding_whitespace_is_transport_not_repair():
    assert parse.classify('\n  {"choice": "IDLE"}\n', OPTIONS)["validity"] == "valid"


def test_an_option_that_does_not_exist_is_counted_as_such():
    out = parse.classify('{"choice": "BRAKE"}', OPTIONS)
    assert out == {"choice": None, "validity": "not_an_option", "detail": "BRAKE"}


def test_a_near_miss_is_not_repaired_into_a_hit():
    """`idle` is not `IDLE`. Repairing it quietly is how benchmarks lie."""
    assert parse.classify('{"choice": "idle"}', OPTIONS)["validity"] == "not_an_option"


def test_json_that_does_not_carry_the_declared_field_is_unparseable():
    """Observed in the pilot: a model in structured mode answered `{"action": ...}`
    against a schema whose only property is `choice`."""
    out = parse.classify('{"action": "IDLE"}', OPTIONS)
    assert out["validity"] == "unparseable"
    assert out["detail"] == "no choice field"


def test_prose_is_unparseable():
    assert parse.classify("I would keep the lane and speed.", OPTIONS)["validity"] == "unparseable"


def test_json_that_is_not_an_object_is_unparseable():
    assert parse.classify('["IDLE"]', OPTIONS)["validity"] == "unparseable"


def test_a_refusal_is_its_own_kind_of_failure():
    out = parse.classify("I'm sorry, I can't help with driving decisions.", OPTIONS)
    assert out["validity"] == "refusal"


def test_an_empty_completion_is_unparseable_not_a_refusal():
    assert parse.classify("", OPTIONS)["validity"] == "unparseable"


def test_a_missing_completion_is_an_api_error():
    """The transport returns None when no content came back at all."""
    assert parse.classify(None, OPTIONS) == {"choice": None, "validity": "api_error", "detail": "no content"}


def test_the_failure_kinds_are_the_four_the_experiment_reports():
    assert parse.KINDS == ("valid", "unparseable", "not_an_option", "refusal", "api_error")

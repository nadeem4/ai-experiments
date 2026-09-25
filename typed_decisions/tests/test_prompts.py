"""The one place the experiment could cheat: the words each model is given.

A decision model is handed `instructions` and a `criteria` map. An LLM has to be
handed the same thing as text. These tests pin that the instruction line and every
option key and description survive verbatim, and that the only thing added is the
transport -- how to answer -- which is identical for every LLM.
"""
from typed_decisions import prompts

INSTRUCTIONS = "You are driving on a highway. Avoid crashing above all, then keep a high speed. Pick the next action."
CRITERIA = {"LANE_LEFT": "Change to the lane on your left.", "IDLE": "Keep your lane and speed."}
STATE = {"your_lane": "rightmost of 4 lanes", "your_speed": "25 m/s (allowed range 20-30)"}


def test_options_are_rendered_key_then_description_in_the_given_order():
    assert prompts.option_block(CRITERIA) == (
        "LANE_LEFT: Change to the lane on your left.\n"
        "IDLE: Keep your lane and speed."
    )


def test_an_option_with_no_description_is_rendered_as_the_bare_key():
    """BANKING77's options carry no descriptions -- `criteria` maps each to None --
    and laya renders those as the key alone. The LLM prompt must do the same or the
    two models are not being asked the same question."""
    assert prompts.option_block({"card arrival": None, "card linking": None}) == "card arrival\ncard linking"


def test_a_dict_state_is_rendered_as_the_field_lines_the_decision_model_sees():
    assert prompts.state_block(STATE) == (
        "your_lane: rightmost of 4 lanes\n"
        "your_speed: 25 m/s (allowed range 20-30)"
    )


def test_a_plain_string_state_is_passed_through_untouched():
    assert prompts.state_block("I am still waiting on my card?") == "I am still waiting on my card?"


def test_the_prompt_carries_the_instructions_and_every_option_verbatim():
    text = prompts.render(INSTRUCTIONS, STATE, CRITERIA)
    assert INSTRUCTIONS in text
    for key, description in CRITERIA.items():
        assert f"{key}: {description}" in text
    assert "rightmost of 4 lanes" in text


def test_the_answer_directive_is_the_same_for_every_model():
    """The directive is transport, not semantics: it tells an LLM how to return an
    answer that a decision model returns by construction. It is a constant, so no
    model gets a differently worded nudge."""
    assert prompts.ANSWER_DIRECTIVE in prompts.render(INSTRUCTIONS, STATE, CRITERIA)
    assert prompts.ANSWER_DIRECTIVE in prompts.render("other", "state", {"a": None})


def test_without_options_is_the_same_prompt_minus_the_option_list():
    """The option-token overhead is measured by differencing the provider's own
    reported prompt tokens, so the two prompts must differ by the option block and
    by nothing else."""
    full = prompts.render(INSTRUCTIONS, STATE, CRITERIA)
    bare = prompts.render(INSTRUCTIONS, STATE, CRITERIA, with_options=False)
    assert prompts.option_block(CRITERIA) in full
    assert prompts.option_block(CRITERIA) not in bare
    assert len(bare) < len(full)
    assert INSTRUCTIONS in bare and "rightmost of 4 lanes" in bare


def test_the_json_schema_pins_the_answer_to_the_option_keys():
    schema = prompts.schema(CRITERIA)
    assert schema["json_schema"]["schema"]["properties"]["choice"]["enum"] == ["LANE_LEFT", "IDLE"]
    assert schema["json_schema"]["strict"] is True


def test_the_option_free_schema_keeps_the_scaffolding_and_drops_only_the_options():
    """The overhead probe has to hold structured mode constant. Some providers bill
    a json_schema as a tool definition, so sending the bare prompt with no schema at
    all would charge the scaffolding to the options -- which is how the first pilot
    read 498 option tokens for five options."""
    bare = prompts.schema(CRITERIA, with_enum=False)
    choice = bare["json_schema"]["schema"]["properties"]["choice"]
    assert choice == {"type": "string"}
    assert bare["json_schema"]["name"] == prompts.schema(CRITERIA)["json_schema"]["name"]
    assert bare["json_schema"]["strict"] is True


# =============================================================================
# Reading the answer back: validity, with no repair
# =============================================================================
#
# Did the call return one of the allowed options, without post-hoc repair?
#
# This is a first-class result, not an error to hide. So the rules are strict on
# purpose: the answer is the `choice` field of a JSON object, spelled exactly as the
# option is spelled. Nothing is fuzzy-matched, nothing is retried, and a model that
# answers with prose or with a key of its own invention has not done the job.


OPTIONS = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]


def test_a_well_formed_answer_is_valid():
    assert prompts.classify('{"choice": "IDLE"}', OPTIONS) == {"choice": "IDLE", "validity": "valid", "detail": None}


def test_surrounding_whitespace_is_transport_not_repair():
    assert prompts.classify('\n  {"choice": "IDLE"}\n', OPTIONS)["validity"] == "valid"


def test_an_option_that_does_not_exist_is_counted_as_such():
    out = prompts.classify('{"choice": "BRAKE"}', OPTIONS)
    assert out == {"choice": None, "validity": "not_an_option", "detail": "BRAKE"}


def test_a_near_miss_is_not_repaired_into_a_hit():
    """`idle` is not `IDLE`. Repairing it quietly is how benchmarks lie."""
    assert prompts.classify('{"choice": "idle"}', OPTIONS)["validity"] == "not_an_option"


def test_json_that_does_not_carry_the_declared_field_is_unparseable():
    """Observed in the pilot: a model in structured mode answered `{"action": ...}`
    against a schema whose only property is `choice`."""
    out = prompts.classify('{"action": "IDLE"}', OPTIONS)
    assert out["validity"] == "unparseable"
    assert out["detail"] == "no choice field"


def test_prose_is_unparseable():
    assert prompts.classify("I would keep the lane and speed.", OPTIONS)["validity"] == "unparseable"


def test_json_that_is_not_an_object_is_unparseable():
    assert prompts.classify('["IDLE"]', OPTIONS)["validity"] == "unparseable"


def test_a_refusal_is_its_own_kind_of_failure():
    out = prompts.classify("I'm sorry, I can't help with driving decisions.", OPTIONS)
    assert out["validity"] == "refusal"


def test_an_empty_completion_is_unparseable_not_a_refusal():
    assert prompts.classify("", OPTIONS)["validity"] == "unparseable"


def test_a_missing_completion_is_an_api_error():
    """The transport returns None when no content came back at all."""
    assert prompts.classify(None, OPTIONS) == {"choice": None, "validity": "api_error", "detail": "no content"}


def test_the_failure_kinds_are_the_four_the_experiment_reports():
    assert prompts.KINDS == ("valid", "unparseable", "not_an_option", "refusal", "api_error")

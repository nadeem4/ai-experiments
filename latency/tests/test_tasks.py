"""Two tasks, five options and seventy-seven, built from data that already exists.

The highway examples are lifted out of the arena's recorded runs and the BANKING77
examples out of the existing harness, so every experiment is looking at the same
states in the same order. Nothing here invents an example.
"""
import pytest

from latency import tasks

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "You are driving on a highway. Pick the next action.",
        "criteria": {"IDLE": "Keep your lane and speed.", "FASTER": "Accelerate."},
    }
}
EVENTS = [
    {"type": "start", "game": "highway", "seed": 3, "questions": QUESTIONS},
    {"type": "step", "t": 1, "state": {"your_lane": "leftmost of 4 lanes"}, "action": "IDLE"},
    {"type": "step", "t": 2, "state": {"your_lane": "lane 2 of 4"}, "action": "FASTER"},
    {"type": "end", "steps": 2},
]


def test_every_step_of_a_recorded_episode_becomes_one_example():
    examples = tasks.highway_examples(EVENTS, seed=3)
    assert [e["state"] for e in examples] == [{"your_lane": "leftmost of 4 lanes"}, {"your_lane": "lane 2 of 4"}]


def test_an_example_id_names_the_episode_and_the_step():
    assert [e["id"] for e in tasks.highway_examples(EVENTS, seed=3)] == ["seed-3:t1", "seed-3:t2"]


def test_the_recorded_answer_is_not_ground_truth():
    """No reference policy ships with the highway game -- only `idle` and `random`
    baselines -- so what Jev happened to pick is another model's opinion, not a
    label. Every highway example is gold-free and the report says agreement, not
    accuracy."""
    assert all(e["gold"] is None for e in tasks.highway_examples(EVENTS, seed=3))


def test_the_question_comes_out_of_the_recording_rather_than_being_retyped():
    instructions, criteria = tasks.highway_question(EVENTS)
    assert instructions == QUESTIONS["action"]["instructions"]
    assert criteria == QUESTIONS["action"]["criteria"]


def test_episodes_that_disagree_about_the_question_are_refused():
    """Every model must be asked the same words. If two recorded episodes carry
    different option text, silently taking the first one would hide that."""
    other = [dict(EVENTS[0], questions={"action": {**QUESTIONS["action"], "instructions": "different"}})]
    with pytest.raises(ValueError):
        tasks.highway_question(EVENTS + other)


def test_examples_are_shuffled_once_with_a_fixed_seed_so_a_limit_is_a_sample():
    """Episodes are recorded in step order, so an unshuffled `--limit 10` would be
    the first ten steps of one episode -- ten near-identical states. One fixed
    permutation, shared by every model, makes a limit a sample."""
    rows = [{"id": f"e{i}"} for i in range(20)]
    first = tasks.shuffled(rows, seed=7)
    assert [r["id"] for r in first] != [r["id"] for r in rows]
    assert [r["id"] for r in tasks.shuffled(rows, seed=7)] == [r["id"] for r in first]


def test_a_banking77_row_becomes_an_example_with_its_benchmark_label_as_gold():
    rows = [{"id": "test-0", "text": "my card has not arrived", "label": "card_arrival"}]
    examples = tasks.banking77_examples(rows)
    assert examples == [{"id": "test-0", "state": "my card has not arrived", "gold": "card arrival"}]


def test_banking77_gold_is_the_option_text_so_it_compares_to_what_a_model_returns():
    """Options are the labels with underscores replaced by spaces -- BANKING77's
    frozen option texts -- so gold has to be in the same alphabet as the answer."""
    rows = [{"id": "test-1", "text": "x", "label": "lost_or_stolen_card"}]
    assert tasks.banking77_examples(rows)[0]["gold"] == "lost or stolen card"


def test_the_two_tasks_are_the_two_option_counts_the_experiment_contrasts():
    assert tasks.NAMES == ("highway", "banking77")

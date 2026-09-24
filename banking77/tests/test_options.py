import json

from banking77 import options

LABELS = [
    "Refund_not_showing_up",
    "activate_my_card",
    "age_limit",
    "card_arrival",
]


def test_option_text_replaces_underscores_and_nothing_else():
    assert options.option_text("top_up_by_cash_or_cheque") == "top up by cash or cheque"
    assert options.option_text("reverted_card_payment?") == "reverted card payment?"
    assert options.option_text("Refund_not_showing_up") == "Refund not showing up"


def test_option_texts_keep_label_order_and_stay_unique():
    texts = options.option_texts(LABELS)
    assert texts == ["Refund not showing up", "activate my card", "age limit", "card arrival"]
    assert len(set(texts)) == len(texts)


def test_freeze_then_load_round_trips_the_frozen_order(tmp_path):
    path = tmp_path / "options.json"
    options.freeze(path, LABELS)
    assert options.load(path) == LABELS
    assert json.loads(path.read_text())["labels"] == LABELS


def test_freeze_refuses_to_change_an_already_frozen_file(tmp_path):
    path = tmp_path / "options.json"
    options.freeze(path, LABELS)
    try:
        options.freeze(path, list(reversed(LABELS)))
    except ValueError as e:
        assert "frozen" in str(e)
    else:
        raise AssertionError("re-freezing a different order must raise")


def test_freeze_is_idempotent_for_the_same_order(tmp_path):
    path = tmp_path / "options.json"
    options.freeze(path, LABELS)
    options.freeze(path, LABELS)
    assert options.load(path) == LABELS


def test_criteria_maps_every_option_text_to_no_description():
    assert options.criteria(["a b", "c d"]) == {"a b": None, "c d": None}


def test_coarse_groups_partition_the_full_label_set():
    grouped = [label for labels in options.COARSE_GROUPS.values() for label in labels]
    assert len(grouped) == len(set(grouped)), "a label appears in two groups"
    assert set(grouped) == set(options.ALL_LABELS)


def test_coarse_groups_are_between_eight_and_twelve():
    assert 8 <= len(options.COARSE_GROUPS) <= 12


def test_all_labels_has_the_seventy_seven_banking77_intents():
    assert len(options.ALL_LABELS) == 77
    assert options.ALL_LABELS == sorted(options.ALL_LABELS)


def test_group_of_finds_the_group_holding_a_label():
    for group, labels in options.COARSE_GROUPS.items():
        for label in labels:
            assert options.group_of(label) == group


def test_subset_labels_is_seeded_deterministic_and_nested_by_size():
    ten = options.subset_labels(10, seed=7)
    twenty = options.subset_labels(20, seed=7)
    assert len(ten) == 10 and len(twenty) == 20
    assert set(ten) <= set(twenty), "smaller subsets nest inside larger ones"
    assert ten == options.subset_labels(10, seed=7)
    assert ten == sorted(ten, key=options.ALL_LABELS.index), "subset keeps the frozen option order"


def test_subset_labels_of_the_full_size_is_every_label_in_order():
    assert options.subset_labels(77, seed=7) == options.ALL_LABELS

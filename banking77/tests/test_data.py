from banking77 import data

CSV = (
    '"text","category"\n'
    '"I am still waiting on my card?","card_arrival"\n'
    '"How do I locate my card?","card_arrival"\n'
    '"Why was I charged twice?","transaction_charged_twice"\n'
)


def test_parse_csv_reads_text_and_label():
    rows = data.parse_csv(CSV)
    assert rows == [
        {"text": "I am still waiting on my card?", "label": "card_arrival"},
        {"text": "How do I locate my card?", "label": "card_arrival"},
        {"text": "Why was I charged twice?", "label": "transaction_charged_twice"},
    ]


def test_parse_csv_rejects_an_unexpected_header():
    try:
        data.parse_csv('"sentence","intent"\n"a","b"\n')
    except ValueError as e:
        assert "header" in str(e)
    else:
        raise AssertionError("an unexpected header must raise")


def test_labels_are_sorted_and_deduplicated():
    assert data.labels_of(data.parse_csv(CSV)) == ["card_arrival", "transaction_charged_twice"]


def _rows(per_label=10, n_labels=5):
    return [
        {"text": f"text {label} {i}", "label": f"label_{label}"}
        for label in range(n_labels)
        for i in range(per_label)
    ]


def test_stratified_split_holds_out_the_requested_fraction_of_every_label():
    train, val = data.stratified_split(_rows(10, 5), val_frac=0.1, seed=0)
    assert len(val) == 5 and len(train) == 45
    assert sorted(r["label"] for r in val) == [f"label_{i}" for i in range(5)]


def test_stratified_split_is_a_partition():
    rows = _rows(7, 4)
    train, val = data.stratified_split(rows, val_frac=0.1, seed=0)
    assert len(train) + len(val) == len(rows)
    assert not [r for r in train if r in val]
    assert sorted(map(repr, train + val)) == sorted(map(repr, rows))


def test_stratified_split_is_deterministic_for_a_seed_and_moves_with_it():
    a, _ = data.stratified_split(_rows(10, 5), val_frac=0.2, seed=1)
    b, _ = data.stratified_split(_rows(10, 5), val_frac=0.2, seed=1)
    c, _ = data.stratified_split(_rows(10, 5), val_frac=0.2, seed=2)
    assert a == b
    assert a != c


def test_stratified_split_keeps_at_least_one_example_per_label_in_validation():
    train, val = data.stratified_split(_rows(3, 4), val_frac=0.1, seed=0)
    assert len({r["label"] for r in val}) == 4


def test_shuffled_is_a_permutation_not_a_filter():
    rows = _rows(4, 5)
    assert sorted(map(repr, data.shuffled(rows, seed=3))) == sorted(map(repr, rows))


def test_shuffled_is_the_same_order_for_every_arm():
    rows = _rows(4, 5)
    assert data.shuffled(rows, seed=3) == data.shuffled(rows, seed=3)
    assert data.shuffled(rows, seed=3) != rows


def test_shuffled_spreads_the_labels_so_a_limit_is_a_sample_not_a_prefix():
    # BANKING77's test split is ordered by label: the first 200 rows hold 5 intents
    # of 77, so an unshuffled --limit would measure five labels and call it accuracy.
    rows = _rows(40, 20)
    assert len({r["label"] for r in rows[:40]}) == 1
    assert len({r["label"] for r in data.shuffled(rows, seed=3)[:40]}) >= 10


def test_example_ids_are_stable_and_unique():
    rows = data.with_ids(data.parse_csv(CSV), split="test")
    assert [r["id"] for r in rows] == ["test-0", "test-1", "test-2"]

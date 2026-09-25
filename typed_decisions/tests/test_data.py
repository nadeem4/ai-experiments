"""The two loaders, tested without the network.

Everything that could silently produce a wrong number is here: the label names
have to come out of the file rather than out of my memory, the ids have to be
stable, and the shuffle has to happen before any limit. CLINC's test split ships
sorted by intent, so the last one is not a style preference -- an unshuffled
`--limit 50` on it is fifty rows of one intent.
"""
import json

import pyarrow as pa
import pytest

from typed_decisions import data


def table(texts, labels, names, text_column="text", label_column="label"):
    """A parquet table shaped like the ones on the hub: an int column plus a
    `huggingface` schema-metadata blob naming the classes."""
    info = {"info": {"features": {
        text_column: {"dtype": "string", "_type": "Value"},
        label_column: {"names": list(names), "_type": "ClassLabel"},
    }}}
    schema = pa.schema(
        [pa.field(text_column, pa.string()), pa.field(label_column, pa.int64())],
        metadata={b"huggingface": json.dumps(info).encode()},
    )
    return pa.Table.from_pydict({text_column: list(texts), label_column: list(labels)}, schema=schema)


class TestLabelNames:
    def test_reads_the_class_names_out_of_the_file(self):
        t = table(["a", "b"], [0, 1], ["World", "Sports"])
        assert data.label_names(t, "label") == ["World", "Sports"]

    def test_refuses_a_column_that_is_not_a_class_label(self):
        t = table(["a"], [0], ["World"])
        with pytest.raises(ValueError, match="text"):
            data.label_names(t, "text")

    def test_refuses_a_file_with_no_huggingface_metadata(self):
        t = pa.Table.from_pydict({"text": ["a"], "label": [0]})
        with pytest.raises(ValueError, match="metadata"):
            data.label_names(t, "label")


class TestRows:
    def test_label_is_the_name_not_the_integer(self):
        t = table(["hello", "goodbye"], [1, 0], ["World", "Sports"])
        rows = data.rows_from_table(t, "test", "text", "label")
        assert [r["label"] for r in rows] == ["Sports", "World"]

    def test_ids_are_positional_and_stable(self):
        t = table(["a", "b", "c"], [0, 0, 0], ["World"])
        rows = data.rows_from_table(t, "test", "text", "label")
        assert [r["id"] for r in rows] == ["test-0", "test-1", "test-2"]

    def test_the_id_survives_shuffling(self):
        """The id is the resume key, so it must name the row in the file and not
        the row's position after an arm reordered it."""
        t = table(list("abcdefgh"), [0] * 8, ["World"])
        rows = data.rows_from_table(t, "test", "text", "label")
        by_text = {r["text"]: r["id"] for r in rows}
        for r in data.shuffled(rows, seed=3):
            assert by_text[r["text"]] == r["id"]


class TestShuffle:
    def test_the_same_seed_gives_the_same_order(self):
        rows = [{"id": i} for i in range(50)]
        assert data.shuffled(rows, 0) == data.shuffled(rows, 0)

    def test_a_different_seed_gives_a_different_order(self):
        rows = [{"id": i} for i in range(50)]
        assert data.shuffled(rows, 0) != data.shuffled(rows, 1)

    def test_it_does_not_mutate_its_argument(self):
        rows = [{"id": i} for i in range(20)]
        data.shuffled(rows, 0)
        assert [r["id"] for r in rows] == list(range(20))

    def test_shuffling_before_a_limit_is_what_makes_a_limit_a_sample(self):
        """The regression this exists for: a label-sorted split, limited without
        a shuffle, measures one class and looks entirely plausible."""
        # CLINC150's test split, in shape: 151 intents, 30 rows each, sorted.
        sorted_by_label = [{"id": i, "label": f"intent-{i // 30}"} for i in range(151 * 30)]
        unshuffled = {r["label"] for r in sorted_by_label[:50]}
        sampled = {r["label"] for r in data.shuffled(sorted_by_label, 0)[:50]}
        assert len(unshuffled) == 2, "the raw order gives 50 rows of two intents"
        # 50 draws from 151 intents collide a little, so ~43 distinct, not 50.
        assert len(sampled) >= 40, "the shuffled order spreads 50 rows across ~43 intents"


class TestValidationSplit:
    def test_validation_comes_out_of_train_and_the_two_do_not_overlap(self):
        rows = [{"id": f"train-{i}", "label": f"L{i % 5}"} for i in range(200)]
        train, val = data.stratified_split(rows, val_frac=0.1, seed=0)
        assert len(train) + len(val) == 200
        assert not {r["id"] for r in train} & {r["id"] for r in val}

    def test_every_label_is_represented_in_the_validation_split(self):
        """A temperature fitted on a split missing a label is fitted on a
        different problem."""
        rows = [{"id": f"train-{i}", "label": f"L{i % 20}"} for i in range(100)]
        _, val = data.stratified_split(rows, val_frac=0.05, seed=0)
        assert len({r["label"] for r in val}) == 20

    def test_it_is_deterministic(self):
        rows = [{"id": f"train-{i}", "label": f"L{i % 5}"} for i in range(100)]
        assert data.stratified_split(rows, 0.1, 0)[1] == data.stratified_split(rows, 0.1, 0)[1]


class TestDatasetDefinitions:
    def test_both_datasets_are_declared_with_a_licence(self):
        assert set(data.DATASETS) == {"ag_news", "clinc150"}
        for spec in data.DATASETS.values():
            assert spec["licence"]
            assert spec["repo"]

    def test_clinc_names_the_one_config_it_uses(self):
        """Three configs share one test split; the protocol has to say which one
        the train and validation rows came from."""
        assert data.DATASETS["clinc150"]["config"] == "plus"
        assert all(f.startswith("plus/") for f in data.DATASETS["clinc150"]["files"].values())

    def test_ag_news_licence_is_recorded_as_unknown_rather_than_guessed(self):
        assert "unknown" in data.DATASETS["ag_news"]["licence"].lower()

"""Two tasks, four options and a hundred and fifty-one, owned by this experiment.

The tests that matter here are the ones about coupling and about order: this
module must not reach into another experiment's data, and it must shuffle before
it limits.
"""
import json

import pyarrow as pa
import pytest

from typed_decisions import spec, tasks


def rows(n, labels):
    return [{"id": f"test-{i}", "split": "test", "text": f"row {i}", "label": labels[i % len(labels)]}
            for i in range(n)]


AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


class TestSelfContained:
    def test_the_module_imports_nothing_from_another_experiment(self):
        """It used to do `from banking77 import data, options`. That experiment
        deliberately varies its option texts as part of its method, so importing
        from it meant this experiment's task changed whenever that one ran an
        arm. One experiment, one directory."""
        source = (pytest.importorskip("pathlib").Path(tasks.__file__)).read_text(encoding="utf-8")
        for sibling in ("banking77", "rerank", "arena", "latency", "decision_cost"):
            assert f"import {sibling}" not in source
            assert f"from {sibling}" not in source

    def test_the_two_tasks_are_the_new_ones(self):
        assert tasks.NAMES == ("ag_news", "clinc150")


class TestOptionTexts:
    def test_a_snake_case_label_gets_a_readable_description(self):
        assert tasks.describe("oil_change_how") == "oil change how"

    def test_a_label_that_is_already_words_gets_no_description(self):
        """`prompts.py` renders a `None` description as the bare key, which is
        exactly how laya renders it, so the two transports still match."""
        assert tasks.describe("Sports") is None

    def test_the_option_key_is_the_dataset_label_so_gold_needs_no_mapping(self):
        texts = tasks.option_texts(["World", "oil_change_how"])
        assert list(texts) == ["World", "oil_change_how"]

    def test_the_canonical_order_is_the_datasets_own_order(self):
        """Not alphabetical, not shuffled: it is what the spec pins and what
        every position-bias arm is a permutation of."""
        texts = tasks.option_texts(["Sci/Tech", "World", "Sports"])
        assert list(texts) == ["Sci/Tech", "World", "Sports"]


class TestBuild:
    def test_the_examples_are_shuffled_before_the_limit(self):
        label_sorted = [{"id": f"test-{i}", "label": f"L{i // 30}", "text": "x"}
                        for i in range(151 * 30)]
        task = tasks.build("clinc150", label_sorted, [f"L{i}" for i in range(151)],
                           limit=50, seed=0, bias_subset=5)
        assert len({e["label"] for e in task["examples"]}) > 30

    def test_the_limit_leaves_the_rest_as_a_warm_up_pool(self):
        """Warm-ups come from outside the measured window, so provider-side
        prompt caching cannot make a measured call look faster or cheaper."""
        task = tasks.build("ag_news", rows(100, AG_LABELS), AG_LABELS, limit=10, bias_subset=2)
        assert len(task["examples"]) == 10
        assert len(task["warmup_pool"]) == 90
        assert not ({e["id"] for e in task["examples"]}
                    & {e["id"] for e in task["warmup_pool"]})

    def test_a_limit_of_zero_is_everything(self):
        task = tasks.build("ag_news", rows(20, AG_LABELS), AG_LABELS, limit=0, bias_subset=2)
        assert len(task["examples"]) == 20
        assert task["warmup_pool"] == []

    def test_the_task_carries_a_frozen_spec_whose_hash_checks_out(self):
        task = tasks.build("ag_news", rows(20, AG_LABELS), AG_LABELS, limit=5, bias_subset=2)
        assert spec.hash_of(task["spec"]["payload"]) == task["spec"]["hash"]

    def test_the_spec_records_the_dataset_and_the_config(self):
        task = tasks.build("clinc150", rows(20, ["a", "b"]), ["a", "b"], limit=5, bias_subset=2)
        assert task["spec"]["payload"]["dataset"] == "clinc/clinc_oos"
        assert task["spec"]["payload"]["config"] == "plus"

    def test_rebuilding_the_same_task_gives_the_same_hash(self):
        one = tasks.build("ag_news", rows(50, AG_LABELS), AG_LABELS, limit=10, seed=0, bias_subset=4)
        two = tasks.build("ag_news", rows(50, AG_LABELS), AG_LABELS, limit=10, seed=0, bias_subset=4)
        assert one["spec"]["hash"] == two["spec"]["hash"]

    def test_a_different_seed_gives_a_different_hash(self):
        one = tasks.build("ag_news", rows(50, AG_LABELS), AG_LABELS, limit=10, seed=0, bias_subset=4)
        two = tasks.build("ag_news", rows(50, AG_LABELS), AG_LABELS, limit=10, seed=1, bias_subset=4)
        assert one["spec"]["hash"] != two["spec"]["hash"]

    def test_the_bias_subset_never_exceeds_the_examples_available(self):
        task = tasks.build("ag_news", rows(6, AG_LABELS), AG_LABELS, limit=3, bias_subset=40)
        assert task["spec"]["payload"]["position_bias"]["subset_size"] == 3

    def test_every_task_declares_its_licence_and_its_source(self):
        task = tasks.build("ag_news", rows(6, AG_LABELS), AG_LABELS, limit=3, bias_subset=2)
        assert "unknown" in task["licence"].lower()
        assert "fancyzhx/ag_news" in task["source"]

    def test_an_unknown_task_is_refused(self):
        with pytest.raises(SystemExit, match="banking77"):
            tasks.build("banking77", rows(4, AG_LABELS), AG_LABELS)


class TestInstructions:
    def test_each_task_has_one_instruction_line_and_it_is_in_the_spec(self):
        task = tasks.build("ag_news", rows(6, AG_LABELS), AG_LABELS, limit=3, bias_subset=2)
        assert task["spec"]["payload"]["instructions"] == tasks.INSTRUCTIONS["ag_news"]

    def test_clinc_tells_the_model_what_the_out_of_scope_label_is_for(self):
        """151 labels, one of which is `oos`. A model not told what it means is
        being asked a different, harder question than the benchmark defines."""
        assert "oos" in tasks.INSTRUCTIONS["clinc150"]


# =============================================================================
# The loaders: where the two datasets come from
# =============================================================================
#
# The two loaders, tested without the network.
#
# Everything that could silently produce a wrong number is here: the label names
# have to come out of the file rather than out of my memory, the ids have to be
# stable, and the shuffle has to happen before any limit. CLINC's test split ships
# sorted by intent, so the last one is not a style preference -- an unshuffled
# `--limit 50` on it is fifty rows of one intent.


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
        assert tasks.label_names(t, "label") == ["World", "Sports"]

    def test_refuses_a_column_that_is_not_a_class_label(self):
        t = table(["a"], [0], ["World"])
        with pytest.raises(ValueError, match="text"):
            tasks.label_names(t, "text")

    def test_refuses_a_file_with_no_huggingface_metadata(self):
        t = pa.Table.from_pydict({"text": ["a"], "label": [0]})
        with pytest.raises(ValueError, match="metadata"):
            tasks.label_names(t, "label")


class TestRows:
    def test_label_is_the_name_not_the_integer(self):
        t = table(["hello", "goodbye"], [1, 0], ["World", "Sports"])
        rows = tasks.rows_from_table(t, "test", "text", "label")
        assert [r["label"] for r in rows] == ["Sports", "World"]

    def test_ids_are_positional_and_stable(self):
        t = table(["a", "b", "c"], [0, 0, 0], ["World"])
        rows = tasks.rows_from_table(t, "test", "text", "label")
        assert [r["id"] for r in rows] == ["test-0", "test-1", "test-2"]

    def test_the_id_survives_shuffling(self):
        """The id is the resume key, so it must name the row in the file and not
        the row's position after an arm reordered it."""
        t = table(list("abcdefgh"), [0] * 8, ["World"])
        rows = tasks.rows_from_table(t, "test", "text", "label")
        by_text = {r["text"]: r["id"] for r in rows}
        for r in tasks.shuffled(rows, seed=3):
            assert by_text[r["text"]] == r["id"]


class TestShuffle:
    def test_the_same_seed_gives_the_same_order(self):
        rows = [{"id": i} for i in range(50)]
        assert tasks.shuffled(rows, 0) == tasks.shuffled(rows, 0)

    def test_a_different_seed_gives_a_different_order(self):
        rows = [{"id": i} for i in range(50)]
        assert tasks.shuffled(rows, 0) != tasks.shuffled(rows, 1)

    def test_it_does_not_mutate_its_argument(self):
        rows = [{"id": i} for i in range(20)]
        tasks.shuffled(rows, 0)
        assert [r["id"] for r in rows] == list(range(20))

    def test_shuffling_before_a_limit_is_what_makes_a_limit_a_sample(self):
        """The regression this exists for: a label-sorted split, limited without
        a shuffle, measures one class and looks entirely plausible."""
        # CLINC150's test split, in shape: 151 intents, 30 rows each, sorted.
        sorted_by_label = [{"id": i, "label": f"intent-{i // 30}"} for i in range(151 * 30)]
        unshuffled = {r["label"] for r in sorted_by_label[:50]}
        sampled = {r["label"] for r in tasks.shuffled(sorted_by_label, 0)[:50]}
        assert len(unshuffled) == 2, "the raw order gives 50 rows of two intents"
        # 50 draws from 151 intents collide a little, so ~43 distinct, not 50.
        assert len(sampled) >= 40, "the shuffled order spreads 50 rows across ~43 intents"


class TestValidationSplit:
    def test_validation_comes_out_of_train_and_the_two_do_not_overlap(self):
        rows = [{"id": f"train-{i}", "label": f"L{i % 5}"} for i in range(200)]
        train, val = tasks.stratified_split(rows, val_frac=0.1, seed=0)
        assert len(train) + len(val) == 200
        assert not {r["id"] for r in train} & {r["id"] for r in val}

    def test_every_label_is_represented_in_the_validation_split(self):
        """A temperature fitted on a split missing a label is fitted on a
        different problem."""
        rows = [{"id": f"train-{i}", "label": f"L{i % 20}"} for i in range(100)]
        _, val = tasks.stratified_split(rows, val_frac=0.05, seed=0)
        assert len({r["label"] for r in val}) == 20

    def test_it_is_deterministic(self):
        rows = [{"id": f"train-{i}", "label": f"L{i % 5}"} for i in range(100)]
        assert tasks.stratified_split(rows, 0.1, 0)[1] == tasks.stratified_split(rows, 0.1, 0)[1]


class TestDatasetDefinitions:
    def test_both_datasets_are_declared_with_a_licence(self):
        assert set(tasks.DATASETS) == {"ag_news", "clinc150"}
        for spec in tasks.DATASETS.values():
            assert spec["licence"]
            assert spec["repo"]

    def test_clinc_names_the_one_config_it_uses(self):
        """Three configs share one test split; the protocol has to say which one
        the train and validation rows came from."""
        assert tasks.DATASETS["clinc150"]["config"] == "plus"
        assert all(f.startswith("plus/") for f in tasks.DATASETS["clinc150"]["files"].values())

    def test_ag_news_licence_is_recorded_as_unknown_rather_than_guessed(self):
        assert "unknown" in tasks.DATASETS["ag_news"]["licence"].lower()

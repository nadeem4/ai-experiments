"""Two tasks, four options and a hundred and fifty-one, owned by this experiment.

The tests that matter here are the ones about coupling and about order: this
module must not reach into another experiment's data, and it must shuffle before
it limits.
"""
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

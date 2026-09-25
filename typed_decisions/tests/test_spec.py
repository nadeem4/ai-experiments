"""The frozen run spec and its hash.

The whole point of the spec is that a model measured next month can be compared
against a model measured today *without re-running anything*. That is only sound
if both saw byte-identical inputs, so the spec pins every input and the hash is
the proof. These tests are the proof that the proof works:

  * the hash is stable under re-derivation, or resuming a run would re-ask
    everything;
  * the hash moves when any pinned input moves, or a reworded instruction would
    quietly pass as the same experiment;
  * `diff` names what changed, so a refusal to mix hashes is actionable.
"""
import pytest

from typed_decisions import spec

OPTIONS = {"World": "world news", "Sports": "sport", "Business": "business", "Sci/Tech": "science"}
EXAMPLES = [
    {"id": "test-4", "text": "a", "label": "World"},
    {"id": "test-1", "text": "b", "label": "Sports"},
    {"id": "test-9", "text": "c", "label": "Sci/Tech"},
]


def build(**over):
    kwargs = dict(
        task="ag_news", dataset="fancyzhx/ag_news", config="default", split="test",
        seed=0, examples=EXAMPLES, instructions="Pick the topic.", option_texts=OPTIONS,
        bias_subset=2, n_random_orders=3, validation=[],
    )
    kwargs.update(over)
    return spec.build(**kwargs)


class TestHashStability:
    def test_re_deriving_the_same_spec_gives_the_same_hash(self):
        assert build()["hash"] == build()["hash"]

    def test_the_hash_does_not_depend_on_key_insertion_order(self):
        reordered = {k: OPTIONS[k] for k in reversed(list(OPTIONS))}
        # The option *order* is part of the experiment, so this must differ...
        assert build(option_texts=reordered)["hash"] != build()["hash"]
        # ...but a payload re-serialised with its keys in another order must not.
        payload = build()["payload"]
        shuffled_payload = {k: payload[k] for k in reversed(list(payload))}
        assert spec.hash_of(shuffled_payload) == build()["hash"]

    def test_the_hash_is_recomputable_from_the_payload_alone(self):
        """A spec read back off disk must hash to what it claims, or the stored
        hash is just a label."""
        s = build()
        assert spec.hash_of(s["payload"]) == s["hash"]


class TestHashSensitivity:
    @pytest.mark.parametrize("change", [
        {"instructions": "Pick the topic!"},
        {"seed": 1},
        {"split": "train"},
        {"dataset": "somebody/else"},
        {"config": "small"},
        {"n_random_orders": 4},
        {"bias_subset": 3},
        {"option_texts": dict(OPTIONS, World="global news")},
        {"examples": EXAMPLES[::-1]},
        {"examples": EXAMPLES[:2]},
        {"validation": [{"id": "train-7", "text": "v", "label": "World"}]},
    ])
    def test_any_pinned_input_moving_moves_the_hash(self, change):
        assert build(**change)["hash"] != build()["hash"]

    def test_a_regenerated_example_list_is_caught(self):
        """Same ids, different order -- the failure mode that looks like nothing
        happened and invalidates every paired comparison."""
        reordered = [EXAMPLES[1], EXAMPLES[0], EXAMPLES[2]]
        assert build(examples=reordered)["hash"] != build()["hash"]


class TestExampleList:
    def test_the_spec_carries_the_ordered_example_ids(self):
        assert build()["payload"]["example_ids"] == ["test-4", "test-1", "test-9"]

    def test_the_spec_carries_the_gold_label_of_every_example(self):
        """Position bias needs to know where gold is; reading it back out of the
        dataset at report time would let the dataset move under the spec."""
        assert build()["payload"]["gold"]["test-4"] == "World"


class TestRandomOrders:
    def test_three_orders_per_example_in_the_bias_subset(self):
        orders = build()["payload"]["position_bias"]["random"]
        assert set(orders) == {"test-4", "test-1"}
        assert all(len(v) == 3 for v in orders.values())

    def test_every_order_is_a_permutation_of_all_the_options(self):
        s = build()
        n = len(s["payload"]["canonical_order"])
        for orders in s["payload"]["position_bias"]["random"].values():
            for order in orders:
                assert sorted(order) == list(range(n))

    def test_the_three_orders_of_one_example_are_not_all_the_same(self):
        orders = build()["payload"]["position_bias"]["random"]["test-4"]
        assert len({tuple(o) for o in orders}) > 1

    def test_different_examples_get_different_orders(self):
        orders = build()["payload"]["position_bias"]["random"]
        assert orders["test-4"] != orders["test-1"]

    def test_the_orders_are_in_the_spec_so_every_model_gets_the_identical_ones(self):
        """Not 'generated per model with the same seed' -- literally the same
        list, read off the spec, so an arm cannot drift."""
        assert build()["payload"]["position_bias"]["random"] == \
               build()["payload"]["position_bias"]["random"]


class TestGoldPlacement:
    def test_gold_lands_first_middle_and_last(self):
        s = build()
        canonical = s["payload"]["canonical_order"]
        for example_id, placements in s["payload"]["position_bias"]["gold"].items():
            gold = s["payload"]["gold"][example_id]
            n = len(canonical)
            assert canonical[placements["first"][0]] == gold
            assert canonical[placements["middle"][n // 2]] == gold
            assert canonical[placements["last"][-1]] == gold

    def test_each_placement_still_shows_every_option(self):
        s = build()
        n = len(s["payload"]["canonical_order"])
        for placements in s["payload"]["position_bias"]["gold"].values():
            for order in placements.values():
                assert sorted(order) == list(range(n))

    def test_the_three_placements_differ(self):
        placements = build()["payload"]["position_bias"]["gold"]["test-4"]
        assert len({tuple(v) for v in placements.values()}) == 3


class TestOrderingLookup:
    def test_the_main_arm_is_the_canonical_order_for_every_example(self):
        s = build()
        assert spec.order_for(s, "main", "test-9") == \
               list(range(len(s["payload"]["canonical_order"])))

    def test_a_bias_arm_reads_its_order_off_the_spec(self):
        s = build()
        assert spec.order_for(s, "rand:1", "test-4") == \
               s["payload"]["position_bias"]["random"]["test-4"][1]
        assert spec.order_for(s, "gold:last", "test-4") == \
               s["payload"]["position_bias"]["gold"]["test-4"]["last"]

    def test_asking_for_an_example_outside_the_bias_subset_is_an_error(self):
        with pytest.raises(KeyError):
            spec.order_for(build(), "rand:0", "test-9")

    def test_options_are_rendered_in_the_arms_own_order(self):
        s = build()
        rendered = spec.options_in_order(s, "gold:first", "test-4")
        assert list(rendered)[0] == "World"
        assert set(rendered) == set(OPTIONS)
        assert rendered["World"] == "world news"


class TestDiff:
    def test_two_identical_specs_differ_in_nothing(self):
        assert spec.diff(build()["payload"], build()["payload"]) == []

    def test_a_reworded_instruction_is_named(self):
        d = spec.diff(build()["payload"], build(instructions="Pick one.")["payload"])
        assert any("instructions" in line for line in d)

    def test_a_changed_example_list_is_named_without_dumping_it(self):
        d = spec.diff(build()["payload"], build(examples=EXAMPLES[:2])["payload"])
        assert any("example_ids" in line for line in d)
        assert all(len(line) < 300 for line in d)

    def test_a_changed_option_text_is_named(self):
        d = spec.diff(build()["payload"], build(option_texts=dict(OPTIONS, World="x"))["payload"])
        assert any("option" in line.lower() for line in d)


class TestArms:
    def test_the_arm_names_cover_the_measured_arm_and_both_bias_arms(self):
        assert spec.arms(build()) == ["main", "rand:0", "rand:1", "rand:2",
                                      "gold:first", "gold:middle", "gold:last"]

    def test_only_the_main_arm_counts_towards_the_headline_numbers(self):
        assert spec.MEASURED_ARM == "main"
        assert all(not spec.is_measured(a) for a in spec.arms(build()) if a != "main")

    def test_examples_for_an_arm_are_the_subset_that_arm_covers(self):
        s = build()
        assert spec.examples_for(s, "main") == ["test-4", "test-1", "test-9"]
        assert spec.examples_for(s, "rand:2") == ["test-4", "test-1"]

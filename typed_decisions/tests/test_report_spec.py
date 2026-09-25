"""The report's side of the frozen run spec, and the position-bias tables.

The design promise is that a model added next month is comparable against today's
numbers without re-running anything. These tests are what make the promise
checkable rather than asserted:

  * a subset report equals the full report restricted to the same models, so
    adding rows for a new model cannot move an existing model's number;
  * records carrying different spec hashes are refused, with an error naming what
    differs, so a reworded instruction fails loudly instead of producing a
    quietly invalid table;
  * paired comparisons run over the intersection and say how big it was.
"""
import pytest

from typed_decisions import report

HASH_A = "a" * 64
HASH_B = "b" * 64
KEY_AG = "gpt/ag_news@openrouter"


def _rec(model="gpt", task="ag_news", arm="main", example="test-1", spec_hash=HASH_A,
         latency=100.0, validity="valid", choice="World", **kw):
    base = {"model": model, "model_id": f"vendor/{model}", "task": task, "arm": arm,
            "spec_hash": spec_hash, "example_id": example, "latency_ms": latency,
            "wall_ms": latency, "input_tokens": 110, "output_tokens": 7, "cost": 1e-05,
            "validity": validity, "choice": choice, "correct": None, "retries": 0,
            "failed": False, "structured": True, "n_options": 4,
            "transport": "openrouter", "repeat": 0}
    base.update(kw)
    return base


def _payload(**over):
    base = {"task": "ag_news", "dataset": "fancyzhx/ag_news", "config": "default",
            "split": "test", "seed": 0, "instructions": "Classify it.",
            "canonical_order": ["World", "Sports"],
            "option_texts": [["World", None], ["Sports", None]],
            "example_ids": ["test-1", "test-2"],
            "gold": {"test-1": "World", "test-2": "Sports"},
            "validation_example_ids": [], "validation_gold": {},
            "position_bias": {"n_random_orders": 3, "subset_size": 1, "subset": ["test-1"],
                              "random": {}, "gold": {}}}
    base.update(over)
    return base


class TestRefusingToMixSpecs:
    def test_one_spec_hash_per_task_is_fine(self):
        assert report.require_one_spec([_rec(), _rec(model="phi")],
                                       {HASH_A: _payload()}) == {"ag_news": HASH_A}

    def test_two_tasks_each_with_their_own_spec_is_the_normal_case(self):
        """One spec per task, by construction: the option list, the instruction
        and the example ids all differ between them. The rule is that every
        record *of one task* shares that task's spec, not that the whole store
        has one hash."""
        records = [_rec(task="ag_news", spec_hash=HASH_A),
                   _rec(task="clinc150", spec_hash=HASH_B)]
        specs = {HASH_A: _payload(), HASH_B: _payload(task="clinc150")}
        assert report.require_one_spec(records, specs) == {"ag_news": HASH_A,
                                                           "clinc150": HASH_B}

    def test_one_task_on_two_specs_is_still_refused(self):
        records = [_rec(task="ag_news", spec_hash=HASH_A),
                   _rec(task="ag_news", model="phi", spec_hash=HASH_B),
                   _rec(task="clinc150", spec_hash="c" * 64)]
        specs = {HASH_A: _payload(), HASH_B: _payload(seed=9),
                 "c" * 64: _payload(task="clinc150")}
        with pytest.raises(SystemExit, match="ag_news"):
            report.require_one_spec(records, specs)

    def test_two_spec_hashes_are_refused(self):
        """A quietly invalid comparison must fail loudly. This is the whole
        safety property of the design."""
        records = [_rec(spec_hash=HASH_A), _rec(model="phi", spec_hash=HASH_B)]
        with pytest.raises(SystemExit) as caught:
            report.require_one_spec(records, {HASH_A: _payload(),
                                              HASH_B: _payload(instructions="Pick one.")})
        assert "instructions" in str(caught.value)

    def test_the_refusal_names_which_models_are_on_which_spec(self):
        records = [_rec(model="gpt", spec_hash=HASH_A), _rec(model="phi", spec_hash=HASH_B)]
        with pytest.raises(SystemExit) as caught:
            report.require_one_spec(records, {HASH_A: _payload(), HASH_B: _payload(seed=1)})
        message = str(caught.value)
        assert "gpt" in message and "phi" in message

    def test_a_record_with_no_spec_hash_is_refused_alongside_one_that_has_it(self):
        """An older record cannot be shown to have seen the same inputs, so it
        cannot be compared. Unknown is not a free pass."""
        records = [_rec(spec_hash=HASH_A), _rec(model="phi", spec_hash=None)]
        with pytest.raises(SystemExit, match="unknown"):
            report.require_one_spec(records, {HASH_A: _payload()})

    def test_a_missing_spec_file_is_named_rather_than_silently_skipped(self):
        with pytest.raises(SystemExit, match="aaaaaaaaaaaa"):
            report.require_one_spec([_rec(spec_hash=HASH_A)], {})

    def test_the_difference_is_reported_even_when_only_the_orderings_moved(self):
        other = _payload()
        other["position_bias"] = dict(other["position_bias"], subset=["test-2"])
        with pytest.raises(SystemExit, match="subset"):
            report.require_one_spec([_rec(spec_hash=HASH_A), _rec(spec_hash=HASH_B)],
                                    {HASH_A: _payload(), HASH_B: other})

    def test_a_reworded_instruction_is_named_in_the_error(self):
        with pytest.raises(SystemExit, match="Classify it"):
            report.require_one_spec([_rec(spec_hash=HASH_A), _rec(spec_hash=HASH_B)],
                                    {HASH_A: _payload(),
                                     HASH_B: _payload(instructions="Pick a topic.")})


class TestSubsetReports:
    def test_a_named_subset_is_computed_over_only_those_models(self):
        records = [_rec(model="gpt"), _rec(model="phi"), _rec(model="jev")]
        summary = report.summarise(records, models=["gpt", "jev"])
        assert {s["model"] for s in summary.values()} == {"gpt", "jev"}

    def test_a_subset_report_equals_the_full_report_restricted_to_those_models(self):
        """The property that makes 'add a model later' sound: adding rows for a
        new model must not move any existing model's number by a hair."""
        records = ([_rec(model="gpt", example=f"e{i}", latency=float(i + 1), correct=i % 2)
                    for i in range(20)]
                   + [_rec(model="phi", example=f"e{i}", latency=float(i * 3 + 1), correct=1)
                      for i in range(20)])
        full = report.summarise(records)
        subset = report.summarise(records, models=["gpt"])
        assert subset == {k: v for k, v in full.items() if v["model"] == "gpt"}

    def test_asking_for_a_model_with_no_records_is_an_error_not_an_empty_row(self):
        with pytest.raises(SystemExit, match="deepseek"):
            report.summarise([_rec(model="gpt")], models=["gpt", "deepseek"])

    def test_no_subset_means_every_model_in_the_store(self):
        records = [_rec(model="gpt"), _rec(model="phi")]
        assert {s["model"] for s in report.summarise(records).values()} == {"gpt", "phi"}


class TestOnlyTheMainArmIsMeasured:
    def test_the_position_bias_arms_are_excluded_from_latency_and_cost(self):
        """Those calls re-ask examples that are already in the main arm, so
        folding them in would weight those examples several times over and drag
        the headline cost with them."""
        records = [_rec(arm="main", latency=10.0, cost=1e-05),
                   _rec(arm="rand:0", latency=9999.0, cost=1.0),
                   _rec(arm="gold:last", latency=9999.0, cost=1.0)]
        summary = report.summarise(records)[KEY_AG]
        assert summary["n"] == 1
        assert summary["cost_usd"] == pytest.approx(1e-05)


class TestProbabilityCapability:
    def test_a_model_that_returned_a_distribution_is_flagged(self):
        records = [_rec(model="jev", probabilities={"World": 0.7, "Sports": 0.3})]
        assert report.summarise(records)["jev/ag_news@openrouter"]["returns_probability"] is True

    def test_an_llm_that_returned_only_a_label_is_flagged_as_not(self):
        """Without a probability you cannot threshold, so 'ask a human when
        unsure' is not on the menu at any price. It is a column, not a
        footnote."""
        assert report.summarise([_rec(model="gpt")])[KEY_AG]["returns_probability"] is False

    def test_the_flag_is_measured_from_the_records_rather_than_declared(self):
        """A model that is supposed to return probabilities and did not must show
        as not, or the column is documentation rather than a measurement."""
        records = [_rec(model="jev", probabilities=None)]
        assert report.summarise(records)["jev/ag_news@openrouter"]["returns_probability"] is False


class TestPositionBias:
    def test_the_flip_rate_is_over_the_three_random_orders(self):
        records = [_rec(model="gpt", arm=f"rand:{i}", example="test-1", choice=c)
                   for i, c in enumerate(["World", "Sports", "World"])]
        out = report.position_bias(records)["gpt/ag_news"]
        assert out["flip"]["rate"] == 1.0 and out["flip"]["n"] == 1

    def test_a_stable_model_has_a_flip_rate_of_zero(self):
        records = [_rec(model="jev", arm=f"rand:{i}", example="test-1", choice="World")
                   for i in range(3)]
        assert report.position_bias(records)["jev/ag_news"]["flip"]["rate"] == 0.0

    def test_accuracy_is_reported_at_each_gold_placement_with_the_spread(self):
        records = [_rec(model="gpt", arm="gold:first", example="e1", correct=1),
                   _rec(model="gpt", arm="gold:middle", example="e1", correct=1),
                   _rec(model="gpt", arm="gold:last", example="e1", correct=0)]
        out = report.position_bias(records)["gpt/ag_news"]
        assert out["gold"]["accuracy"] == {"first": 1.0, "middle": 1.0, "last": 0.0}
        assert out["gold"]["spread"] == 1.0

    def test_the_distribution_of_chosen_positions_is_reported_per_model(self):
        """How much a model moves is not the same question as which positions it
        favours. A model that always answers with whatever is listed first has a
        flip rate near one and a completely characteristic distribution."""
        records = [_rec(model="gpt", arm=f"rand:{i}", example=f"e{i}", choice="World",
                        options_shown=["World", "Sports", "Business", "Sci/Tech"])
                   for i in range(3)]
        out = report.position_bias(records)["gpt/ag_news"]
        assert out["positions"]["mean_normalised"] == 0.0
        assert out["positions"]["n"] == 3

    def test_a_model_that_always_picks_the_last_option_shows_as_one(self):
        records = [_rec(model="gpt", arm=f"rand:{i}", example=f"e{i}", choice="Sci/Tech",
                        options_shown=["World", "Sports", "Business", "Sci/Tech"])
                   for i in range(3)]
        assert report.position_bias(records)["gpt/ag_news"]["positions"]["mean_normalised"] == 1.0

    def test_the_subset_size_is_carried_so_it_can_be_stated_everywhere(self):
        records = [_rec(model="gpt", arm=f"rand:{i}", example="test-1", choice="World")
                   for i in range(3)]
        assert report.position_bias(records)["gpt/ag_news"]["n_examples"] == 1

    def test_a_run_with_no_bias_arms_reports_nothing_rather_than_zero_bias(self):
        assert report.position_bias([_rec(arm="main")]) == {}

    def test_a_model_that_could_not_answer_at_all_is_marked_as_such(self):
        """Laya cannot fit 151 options into its head budget, so every CLINC call
        fails. Its zero accuracy at every gold position is a structural failure,
        not a position effect, and the row has to say so."""
        records = [_rec(model="laya", arm=arm, example="e1", choice=None, correct=0,
                        validity="api_error")
                   for arm in ("rand:0", "rand:1", "rand:2",
                               "gold:first", "gold:middle", "gold:last")]
        out = report.position_bias(records)["laya/ag_news"]
        assert out["n_valid"] == 0
        assert out["flip"]["rate"] is None, "a total failure is not a 100% flip rate"
        assert out["flip"]["n_unusable"] == 1

    def test_a_model_that_answered_reports_how_many_were_usable(self):
        records = [_rec(model="gpt", arm=f"rand:{i}", example="e1", choice="World")
                   for i in range(3)]
        assert report.position_bias(records)["gpt/ag_news"]["n_valid"] == 3

    def test_the_bias_arms_are_kept_apart_per_task(self):
        records = ([_rec(model="gpt", task="ag_news", arm=f"rand:{i}", example="e",
                         choice="World") for i in range(3)]
                   + [_rec(model="gpt", task="clinc150", arm=f"rand:{i}", example="e",
                           choice=["a", "b", "c"][i]) for i in range(3)])
        out = report.position_bias(records)
        assert out["gpt/ag_news"]["flip"]["rate"] == 0.0
        assert out["gpt/clinc150"]["flip"]["rate"] == 1.0


class TestPairedComparisons:
    def test_two_models_are_compared_on_the_examples_both_answered(self):
        records = ([_rec(model="gpt", example=f"e{i}", correct=1) for i in range(10)]
                   + [_rec(model="phi", example=f"e{i}", correct=0) for i in range(6)])
        out = report.paired_accuracy(records, "gpt", "phi", "ag_news")
        assert out["n"] == 6
        assert out["mean_diff"] == pytest.approx(1.0)

    def test_the_count_is_reported_because_the_intersection_can_be_small(self):
        records = [_rec(model="gpt", example="e1", correct=1),
                   _rec(model="phi", example="e2", correct=1)]
        assert report.paired_accuracy(records, "gpt", "phi", "ag_news")["n"] == 0

    def test_only_the_main_arm_feeds_a_paired_comparison(self):
        records = [_rec(model="gpt", example="e1", correct=1),
                   _rec(model="phi", example="e1", correct=0),
                   _rec(model="phi", example="e1", arm="gold:last", correct=1)]
        out = report.paired_accuracy(records, "gpt", "phi", "ag_news")
        assert out["n"] == 1 and out["mean_diff"] == 1.0

    def test_an_interval_that_crosses_zero_is_reported_as_such(self):
        records = ([_rec(model="gpt", example=f"e{i}", correct=i % 2) for i in range(30)]
                   + [_rec(model="phi", example=f"e{i}", correct=i % 2) for i in range(30)])
        out = report.paired_accuracy(records, "gpt", "phi", "ag_news")
        assert out["ci"][0] <= 0 <= out["ci"][1]


class TestCalibrationFromTheStore:
    def _rows(self, model, arm, n, p, gold="World"):
        return [_rec(model=model, arm=arm, example=f"{arm}-{i}", choice="World",
                     correct=int(gold == "World"), gold=gold,
                     probabilities={"World": p, "Sports": 1 - p})
                for i in range(n)]

    def test_only_models_that_returned_probabilities_are_calibrated(self):
        records = (self._rows("jev", "validation", 20, 0.99)
                   + self._rows("jev", "main", 20, 0.99)
                   + [_rec(model="gpt", arm="main", example="e1")])
        out = report.calibrations(records)
        assert set(out) == {"jev/ag_news"}

    def test_the_temperature_is_fitted_on_the_validation_arm_only(self):
        """The validation rows come out of train. Nothing is ever fitted on
        test, and the arm is what keeps the two apart in one store."""
        records = (self._rows("jev", "validation", 20, 0.99, gold="World")
                   + self._rows("jev", "validation", 20, 0.99, gold="Sports")
                   + self._rows("jev", "main", 10, 0.99))
        out = report.calibrations(records)["jev/ag_news"]
        assert out["n_validation"] == 40 and out["n_test"] == 10
        assert out["temperature"] > 1

    def test_a_model_with_no_validation_rows_reports_a_raw_ece_and_no_temperature(self):
        """Better than quietly fitting on test, which is the one thing that must
        never happen."""
        out = report.calibrations(self._rows("jev", "main", 10, 0.9))["jev/ag_news"]
        assert out["ece_raw"] is not None
        assert out["temperature"] is None
        assert out["ece_scaled"] is None

    def test_a_reliability_curve_is_carried_for_the_figure(self):
        records = self._rows("jev", "main", 10, 0.9)
        assert report.calibrations(records)["jev/ag_news"]["reliability"]["accuracy"]

    def test_a_store_with_no_probabilities_calibrates_nothing(self):
        assert report.calibrations([_rec(model="gpt")]) == {}


class TestWhereTheResultsAreWritten:
    def test_a_full_report_writes_under_the_tag(self):
        assert report.results_name("pilot", None) == "pilot"

    def test_a_subset_report_writes_beside_it_rather_than_over_it(self):
        """`report --models jev,laya` must not silently replace the full run's
        results file and its figures with a three-model slice of them."""
        assert report.results_name("pilot", ["jev", "laya"]) == "pilot-jev+laya"

    def test_the_subset_name_does_not_depend_on_the_order_they_were_typed(self):
        assert report.results_name("pilot", ["laya", "jev"]) ==                report.results_name("pilot", ["jev", "laya"])


class TestAFailedCallIsNotAWrongAnswer:
    """A call that never returned cannot be scored. Recording it as correct=0
    conflates "the model was wrong" with "we got no answer", which understates
    accuracy by the failure rate and inflates the paired-comparison denominator.
    The store already holds such records, so the reader must exclude them."""

    def _rows(self):
        base = {"task": "t", "arm": "main", "spec_hash": "h", "pass": 0}
        return [
            {**base, "model": "m", "example_id": "1", "validity": "valid", "correct": 1},
            {**base, "model": "m", "example_id": "2", "validity": "valid", "correct": 0},
            {**base, "model": "m", "example_id": "3", "validity": "api_error", "correct": 0},
        ]

    def test_accuracy_is_over_answered_calls_only(self):
        from typed_decisions import report
        scored = report.correctness(self._rows(), "m", "t")
        assert scored == {"1": 1, "2": 0}, "the api_error row must not be scored"

    def test_an_unusable_answer_is_still_scored_wrong(self):
        """The model answered, just not usably. Dropping it would pay it for failing."""
        from typed_decisions import report
        base = {"task": "t", "arm": "main", "spec_hash": "h", "pass": 0, "model": "m"}
        rows = [{**base, "example_id": "1", "validity": "unparseable", "correct": 0}]
        assert report.correctness(rows, "m", "t") == {"1": 0}

    def test_the_failed_call_does_not_enter_a_paired_comparison(self):
        from typed_decisions import report
        assert len(report.correctness(self._rows(), "m", "t")) == 2


class TestGoldPlacementExcludesFailedCalls:
    """A call that never returned is not evidence about where the gold option sat.

    `correct` is stored as 0 for a failed call, so appending it raw makes an
    outage look like a position effect -- and the gold-placement spread is the
    one number this experiment exists to measure.
    """

    def test_an_api_error_is_not_scored_as_a_miss_at_that_placement(self):
        records = [
            _rec(model="qwen", task="clinc150", arm="gold:first", example="e1", correct=1),
            _rec(model="qwen", task="clinc150", arm="gold:first", example="e2",
                 validity="api_error", choice=None, correct=0),
            _rec(model="qwen", task="clinc150", arm="gold:middle", example="e1", correct=1),
            _rec(model="qwen", task="clinc150", arm="gold:last", example="e1", correct=1),
        ]
        gold = report.position_bias(records)["qwen/clinc150"]["gold"]
        assert gold["accuracy"]["first"] == 1.0, "the failed call must not count as a miss"
        assert gold["n"]["first"] == 1, "and must not inflate the denominator"
        assert gold["spread"] == 0.0


class TestTheResultsFileDoesNotPublishALocalPath:
    """`results/full.json` and the site are public; the machine's directory layout
    is not a measurement and does not belong in either.

    The local model records its weights directory in `model_id`, which is useful
    in the raw store and wrong in a published file. The reporter is the boundary
    where that gets normalised, so old records stay readable without a re-run.
    """

    def test_a_weights_directory_is_replaced_with_a_plain_label(self):
        assert report.public_model_id(r"laya (C:\projects\jev_demo\arena\models\laya)") \
            == "laya (local weights)"
        assert report.public_model_id("laya (/home/me/models/laya)") == "laya (local weights)"

    def test_a_hosted_model_id_is_left_exactly_as_it_is(self):
        for model_id in ("typesafe/jev-1.13-20260917", "openai/gpt-6-luna", "microsoft/phi-4"):
            assert report.public_model_id(model_id) == model_id

    def test_no_summary_row_carries_a_filesystem_path(self):
        rows = [_rec(model="laya", model_id=r"laya (C:\projects\jev_demo\arena\models\laya)",
                     transport="local-cpu", correct=1)]
        for row in report.summarise(rows).values():
            assert ":\\" not in row["model_id"] and not row["model_id"].count("/") > 1


class TestTheGoldPlacementCarriesItsSignificance:
    """The spread alone invites reading a one-example difference as an effect.

    At 40 examples per placement most spreads are noise, so the results file
    carries the paired test beside the spread rather than leaving every reader to
    recompute it -- and leaving the site to publish a number no file contains.
    """

    def _placement_rows(self, model, first_correct, last_correct):
        rows = []
        for i, (f, l) in enumerate(zip(first_correct, last_correct)):
            rows += [
                _rec(model=model, task="clinc150", arm="gold:first", example=f"e{i}", correct=f),
                _rec(model=model, task="clinc150", arm="gold:middle", example=f"e{i}", correct=f),
                _rec(model=model, task="clinc150", arm="gold:last", example=f"e{i}", correct=l),
            ]
        return rows

    def test_a_lopsided_effect_is_reported_as_significant(self):
        rows = self._placement_rows("phi", [1] * 11 + [0] * 2, [0] * 11 + [1] * 2)
        sig = report.position_bias(rows)["phi/clinc150"]["gold"]["significance"]
        assert sig["n_discordant"] == 13
        assert sig["p_value"] == pytest.approx(0.02246, abs=1e-5)
        assert sig["comparison"] == "gold:first vs gold:last"

    def test_agreement_between_the_placements_is_not_an_effect(self):
        rows = self._placement_rows("jev", [1, 1, 0, 1], [1, 1, 0, 1])
        sig = report.position_bias(rows)["jev/clinc150"]["gold"]["significance"]
        assert sig["n_discordant"] == 0 and sig["p_value"] == 1.0

    def test_a_failed_call_is_not_a_discordant_pair(self):
        """Same rule as everywhere else: a call that never returned is not evidence."""
        rows = self._placement_rows("qwen", [1, 1], [1, 1])
        rows += [_rec(model="qwen", task="clinc150", arm="gold:first", example="x",
                      validity="api_error", choice=None, correct=0),
                 _rec(model="qwen", task="clinc150", arm="gold:last", example="x", correct=1)]
        sig = report.position_bias(rows)["qwen/clinc150"]["gold"]["significance"]
        assert sig["n_discordant"] == 0, "the api_error pair must not count as a flip to correct"


class TestNothingPublishedCarriesALocalPath:
    """Patching each site that copies a model id is whack-a-mole: this one is an
    invariant over the whole structure, checked once before it is written."""

    def test_every_model_id_in_the_tree_is_scrubbed(self):
        raw = r"laya (C:\projects\jev_demo\arena\models\laya)"
        results = {
            "summary": {"laya/ag_news": {"model_id": raw}},
            "option_overhead": {"laya/ag_news": {"model_id": raw}},
            "model_resolution": {"laya": {"model_id": raw, "tried": [[raw, True]]}},
            "models": ["laya"],
        }
        out = report.scrub_local_paths(results)
        assert out["summary"]["laya/ag_news"]["model_id"] == "laya (local weights)"
        assert out["option_overhead"]["laya/ag_news"]["model_id"] == "laya (local weights)"
        assert out["model_resolution"]["laya"]["model_id"] == "laya (local weights)"

    def test_it_leaves_hosted_ids_and_other_fields_alone(self):
        results = {"summary": {"jev/ag_news": {"model_id": "typesafe/jev-1.13-20260917",
                                               "note": "runs at C:\somewhere"}}}
        out = report.scrub_local_paths(results)
        row = out["summary"]["jev/ag_news"]
        assert row["model_id"] == "typesafe/jev-1.13-20260917"
        assert row["note"] == "runs at C:\somewhere", "only model_id is normalised"

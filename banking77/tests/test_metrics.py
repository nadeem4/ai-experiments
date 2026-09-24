import math

from banking77 import metrics

LABELS = ["a", "b", "c"]


def test_accuracy_counts_exact_matches():
    assert metrics.accuracy(["a", "b", "c"], ["a", "b", "b"]) == 2 / 3
    assert metrics.accuracy([], []) == 0.0


def test_correctness_is_the_per_example_vector_accuracy_averages():
    assert metrics.correctness(["a", "b"], ["a", "c"]) == [1, 0]


def test_top_k_accuracy_reads_the_whole_distribution():
    probs = [{"a": 0.5, "b": 0.3, "c": 0.2}, {"a": 0.5, "b": 0.3, "c": 0.2}]
    assert metrics.top_k_accuracy(probs, ["b", "c"], k=1) == 0.0
    assert metrics.top_k_accuracy(probs, ["b", "c"], k=2) == 0.5
    assert metrics.top_k_accuracy(probs, ["b", "c"], k=3) == 1.0


def test_top_k_accuracy_breaks_ties_by_the_option_order_not_by_luck():
    probs = [{"a": 0.5, "b": 0.5, "c": 0.0}]
    assert metrics.top_k_accuracy(probs, ["b"], k=1) == 0.0
    assert metrics.top_k_accuracy(probs, ["a"], k=1) == 1.0


def test_macro_f1_averages_over_labels_including_ones_never_predicted():
    # 'c' is never predicted and never correct: its F1 is 0 and it still counts.
    f1 = metrics.macro_f1(pred=["a", "a", "b"], gold=["a", "b", "c"], labels=LABELS)
    # a: tp=1 fp=1 fn=0 -> P=.5 R=1 F1=2/3 ; b: tp=0 fp=1 fn=1 -> 0 ; c: 0
    assert math.isclose(f1, (2 / 3) / 3)


def test_macro_f1_is_one_when_everything_is_right():
    assert metrics.macro_f1(["a", "b", "c"], ["a", "b", "c"], LABELS) == 1.0


def test_ece_is_zero_for_perfectly_calibrated_confidence():
    conf = [0.1] * 10 + [0.9] * 10
    correct = [0] * 9 + [1] + [1] * 9 + [0]
    assert math.isclose(metrics.ece(conf, correct, bins=15), 0.0, abs_tol=1e-9)


def test_ece_reports_the_gap_when_confidence_is_systematically_too_high():
    assert math.isclose(metrics.ece([1.0] * 4, [1, 0, 0, 0], bins=15), 0.75)


def test_ece_uses_equal_width_bins_and_ignores_empty_ones():
    # two bins occupied, weighted by count
    conf = [0.1, 0.9, 0.9]
    correct = [1, 1, 1]
    assert math.isclose(metrics.ece(conf, correct, bins=15), (1 * 0.9 + 2 * 0.1) / 3)


def test_bootstrap_ci_brackets_the_mean_and_is_seeded():
    values = [1] * 50 + [0] * 50
    low, high = metrics.bootstrap_ci(values, n=1000, seed=0)
    assert low < 0.5 < high
    assert metrics.bootstrap_ci(values, n=1000, seed=0) == (low, high)
    assert metrics.bootstrap_ci(values, n=1000, seed=1) != (low, high)


def test_bootstrap_ci_of_a_constant_is_a_point():
    assert metrics.bootstrap_ci([1] * 20, n=200, seed=0) == (1.0, 1.0)


def test_fit_temperature_finds_one_for_overconfident_logits():
    # gold is right half the time but the logits are extreme: T should rise above 1.
    logits = [[4.0, 0.0], [4.0, 0.0], [0.0, 4.0], [0.0, 4.0]]
    gold = [0, 1, 0, 1]
    assert metrics.fit_temperature(logits, gold) > 1.0


def test_fit_temperature_leaves_confident_and_correct_logits_alone():
    logits = [[4.0, 0.0], [4.0, 0.0], [0.0, 4.0]]
    gold = [0, 0, 1]
    assert metrics.fit_temperature(logits, gold) <= 1.0


def test_apply_temperature_flattens_a_distribution_without_reordering_it():
    p = metrics.apply_temperature([2.0, 1.0, 0.0], 2.0)
    assert math.isclose(sum(p), 1.0)
    assert p[0] > p[1] > p[2]
    assert p[0] < metrics.apply_temperature([2.0, 1.0, 0.0], 1.0)[0]


def test_logits_from_probabilities_survive_a_softmax_round_trip():
    p = [0.7, 0.2, 0.1]
    assert [round(v, 6) for v in metrics.apply_temperature(metrics.to_logits(p), 1.0)] == p


def test_logits_from_probabilities_clip_an_exact_zero():
    assert all(math.isfinite(z) for z in metrics.to_logits([1.0, 0.0]))


def test_mcnemar_counts_the_discordant_pairs():
    r = metrics.mcnemar([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert r["n01"] == 1 and r["n10"] == 1


def test_mcnemar_is_insignificant_when_the_two_arms_agree():
    r = metrics.mcnemar([1, 0, 1, 0], [1, 0, 1, 0])
    assert r["n01"] == r["n10"] == 0
    assert r["p_value"] == 1.0


def test_mcnemar_is_significant_when_one_arm_wins_every_disagreement():
    r = metrics.mcnemar([1] * 20 + [0] * 5, [0] * 20 + [0] * 5)
    assert r["n10"] == 20 and r["n01"] == 0
    assert r["p_value"] < 1e-5


def test_mcnemar_rejects_unequal_arms():
    try:
        metrics.mcnemar([1, 0], [1])
    except ValueError:
        pass
    else:
        raise AssertionError("arms of different length must raise")

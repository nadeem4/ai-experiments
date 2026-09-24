from banking77 import budget

# [CLS] head [SEP] (M a b) (M a b) (M c) [SEP] state [SEP], with MASK=9 and SEP=2.
IDS = [0, 5, 6, 2, 9, 10, 11, 9, 10, 11, 9, 12, 2, 20, 21, 2]
MARKERS = [4, 7, 10]


def test_option_spans_cuts_one_span_per_marker():
    assert budget.option_spans(IDS, MARKERS, sep_id=2) == [[9, 10, 11], [9, 10, 11], [9, 12]]


def test_option_spans_stops_the_last_option_at_the_separator():
    spans = budget.option_spans(IDS, MARKERS, sep_id=2)
    assert 2 not in spans[-1]


def test_option_spans_handles_a_single_option():
    assert budget.option_spans([0, 2, 9, 7, 2], [2], sep_id=2) == [[9, 7]]


def test_option_spans_rejects_markers_that_fell_outside_the_sequence():
    try:
        budget.option_spans(IDS, [4, 7, 99], sep_id=2)
    except ValueError as e:
        assert "marker" in str(e)
    else:
        raise AssertionError("a marker past the end of the sequence must raise")


def test_truncation_stats_counts_nothing_when_every_option_survives():
    stats = budget.truncation_stats(
        spans=[[9, 1, 2], [9, 3, 4]], full_lengths=[3, 3], decoded=["alpha", "beta"]
    )
    assert stats["n_options"] == 2
    assert stats["mean_text_tokens_per_option"] == 2.0
    assert stats["truncated_fraction"] == 0.0
    assert stats["identical_pairs"] == 0
    assert stats["distinct_options"] == 2


def test_truncation_stats_reports_the_fraction_actually_cut():
    stats = budget.truncation_stats(
        spans=[[9, 1], [9, 3, 4], [9, 5]], full_lengths=[4, 3, 6], decoded=["a", "b", "c"]
    )
    assert stats["truncated_fraction"] == 2 / 3


def test_truncation_stats_counts_pairs_that_collapse_to_the_same_string():
    stats = budget.truncation_stats(
        spans=[[9, 1], [9, 1], [9, 1], [9, 2]],
        full_lengths=[5, 5, 5, 5],
        decoded=["top up", "top up", "top up", "card"],
    )
    assert stats["identical_pairs"] == 3  # three of the four labels collapse: 3 choose 2
    assert stats["distinct_options"] == 2
    assert stats["collided_options"] == 3


def test_truncation_stats_rejects_mismatched_inputs():
    try:
        budget.truncation_stats(spans=[[9, 1]], full_lengths=[2, 2], decoded=["a"])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched lengths must raise")


def test_tokens_per_option_matches_layas_own_arithmetic():
    # laya.common.build_sequence: when the options do not fit, every option is cut to
    # max(4, (head_max_len - 16) // n_options) tokens, the [MASK] marker included.
    assert budget.laya_tokens_per_option(head_max_len=256, n_options=77) == 4
    assert budget.laya_tokens_per_option(head_max_len=384, n_options=77) == 4
    assert budget.laya_tokens_per_option(head_max_len=512, n_options=77) == 6
    assert budget.laya_tokens_per_option(head_max_len=768, n_options=77) == 9


def test_the_floor_of_four_makes_192_and_256_the_same_budget_at_77_options():
    # (256-16)//77 is 3, below laya's floor of 4, so the English checkpoint's 192 and
    # the multilingual 256 hand 77 options exactly the same room. The model card's
    # "192 on English, 256 on multilingual" is not a difference at this option count.
    assert budget.laya_tokens_per_option(192, 77) == budget.laya_tokens_per_option(256, 77) == 4


def test_the_floor_of_four_can_overrun_the_head_budget_it_was_meant_to_respect():
    # 77 options x 4 tokens is 308, which does not fit in head_max_len=192 at all.
    assert 77 * budget.laya_tokens_per_option(192, 77) > 192

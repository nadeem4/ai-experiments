"""The match rule and the speedup at the match, on handmade records."""
import pytest

from emeans import tables


def rec(dataset, arm, setting, seed, mse=None, seconds=None):
    r = {"part": "a", "dataset": dataset, "arm": arm, "setting": setting, "seed": seed}
    if mse is not None:
        r["test_mse"] = mse
    if seconds is not None:
        r["train_seconds"] = seconds
    return r


def lloyd(dataset="sift", mse=(10.0, 10.0), seconds=(100.0, 100.0)):
    return [rec(dataset, "lloyd", "iters=25", s, mse[s], seconds[s]) for s in range(len(mse))]


def test_ratios_pair_by_seed_not_by_order():
    records = lloyd(mse=(10.0, 20.0)) + [rec("sift", "emeans", "epochs=3", 1, 40.0),
                                         rec("sift", "emeans", "epochs=3", 0, 5.0)]
    assert tables.mse_ratios(records, "sift", "emeans", "epochs=3") == {0: 0.5, 1: 2.0}


def test_ratios_ignore_other_datasets():
    records = lloyd("sift") + lloyd("arxiv", mse=(1.0, 1.0)) + [
        rec("sift", "emeans", "epochs=3", 0, 10.0), rec("sift", "emeans", "epochs=3", 1, 10.0),
        rec("arxiv", "emeans", "epochs=3", 0, 9.0), rec("arxiv", "emeans", "epochs=3", 1, 9.0)]
    assert tables.mse_ratios(records, "sift", "emeans", "epochs=3") == {0: 1.0, 1: 1.0}


def test_matches_is_mean_ratio_at_most_one():
    assert tables.matches({0: 1.0, 1: 1.0}) is True
    assert tables.matches({0: 0.9, 1: 1.1}) is True
    assert tables.matches({0: 1.0, 1: 1.0002}) is False
    assert tables.matches({}) is False


def worked():
    records = lloyd(mse=(10.0, 10.0), seconds=(100.0, 200.0))
    for e, mse in ((1, 12.0), (2, 10.5), (3, 9.9), (5, 9.5)):
        records += [rec("sift", "emeans", f"epochs={e}", s, mse, 10.0 * e + 10 * s) for s in (0, 1)]
    return records


def test_epochs_to_match_is_the_smallest_matching_count():
    assert tables.epochs_to_match(worked(), "sift", "emeans") == 3


def test_speedup_is_lloyd_time_over_arm_time_paired_by_seed():
    # seed 0: 100 / 30, seed 1: 200 / 40 -> mean of 3.33.. and 5.0
    assert tables.speedup(worked(), "sift", "emeans") == pytest.approx((100 / 30 + 200 / 40) / 2)


def test_unmatched_has_no_epochs_and_no_speedup():
    records = lloyd() + [rec("sift", "minibatch", f"epochs={e}", s, 11.0, 5.0)
                         for e in (1, 2, 3, 5) for s in (0, 1)]
    assert tables.epochs_to_match(records, "sift", "minibatch") is None
    assert tables.speedup(records, "sift", "minibatch") is None

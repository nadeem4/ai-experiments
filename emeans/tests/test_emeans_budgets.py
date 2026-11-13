"""Wall-clock budgets and calibration."""
import numpy as np

from emeans import arms, budgets


def test_epochs_for_budget_floors_with_a_minimum_of_one():
    assert budgets.epochs_for_budget(2.0, 10) == 5
    assert budgets.epochs_for_budget(20.0, 10) == 1
    assert budgets.epochs_for_budget(0.5, 120) == 240
    assert budgets.epochs_for_budget(3.0, 10) == 3


def test_overshoot_is_more_than_ten_percent_over():
    assert budgets.overshoot(11.1, 10) is True
    assert budgets.overshoot(11.0, 10) is False
    assert budgets.overshoot(8, 10) is False


def test_calibrate_gives_one_positive_entry_per_arm():
    rng = np.random.default_rng(0)
    x = rng.random((3000, 4), dtype=np.float32)
    init = arms.random_init(x, 8, 0)
    fns = {a: (lambda x, k, e, i, s, a=a: arms.train(a, x, k, e, i, s))
           for a in ("lloyd", "emeans", "minibatch")}
    cal = budgets.calibrate(fns, x, 8, init)
    assert set(cal) == set(fns) and all(v > 0 for v in cal.values())

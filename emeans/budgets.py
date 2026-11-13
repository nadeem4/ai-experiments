"""Fixed wall-clock budgets: how many epochs fit, and when one overshoots.

Assumption, stated once: training time is linear in the number of epochs (for
Lloyd, iterations), so one measured epoch at the working scale sets the count.
"""
import math


def epochs_for_budget(seconds_per_epoch, budget_s):
    """The largest epoch count whose estimated time fits, and at least 1. A
    budget smaller than one epoch still gets 1; `overshoot` flags it."""
    return max(1, math.floor(budget_s / seconds_per_epoch))


def overshoot(actual_s, budget_s, tolerance=0.10):
    """True when the measured time exceeds the budget by more than `tolerance`.
    Reported next to the budget, never silently clipped."""
    return actual_s > budget_s * (1 + tolerance)


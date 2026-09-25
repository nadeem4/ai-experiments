"""Every figure in this experiment, drawn by code from the results file.

Nothing here is hand-drawn and nothing takes its own input: the charts and the
tables both read `banking77/results/<tag>.json`, so a figure cannot drift away
from the numbers printed beside it. `report.py` calls `write_all` by default, so
the images are regenerated whenever the tables are.

**A figure with nothing to show raises.** A budget panel drawn from a run that
only has the default arm would be a single bar implying a comparison that was
never made. `write_all` catches the refusal, writes no file, and reports the skip
with its reason.

## Styling

Plain, and greyscale on purpose. Meaning is never encoded in colour alone: every
series carries a marker shape or a hatch as well as a shade, and values are
labelled directly rather than through a legend, so the figures survive a
grayscale print and a colour-blind reader.

## What is drawn, and why each one earns its place

  * **budget-cliff** -- accuracy against `head_max_len`. The question the
    experiment asks. Drawn as steps rather than a line because 384, 512 and 768
    are identical to every decimal: a line through three equal points invites the
    reader to extrapolate a slope that is not there.
  * **option-count** -- accuracy against how many options are on the table, with
    the budget held at its default. The answer the experiment did not expect:
    accuracy falls as options are added while nothing is being truncated.
  * **arms** -- every **77-option** arm against the two published numbers. Shows
    what survives fixing the budget, which is most of the gap. The K sweep is
    deliberately not on it: a ten-option arm and a seventy-seven-option arm on
    one accuracy axis make the shorter task look like the better model.
"""
from pathlib import Path

DPI = 160
FIGSIZE = (9.0, 5.0)

SHADES = ("#111111", "#555555", "#888888", "#aaaaaa")
GRID = {"color": "#cccccc", "linewidth": 0.6, "alpha": 0.9}

# The default the model ships with, which every raised-budget arm is measured
# against. Named rather than repeated so the panels cannot disagree about it.
DEFAULT_BUDGET = 256


class NothingToPlot(ValueError):
    """The arms this figure compares are absent from the results.

    Raised rather than drawing a chart with one bar in it. A published figure is
    read as a claim, and a figure missing its comparison is a claim about a
    comparison that never ran."""


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, **GRID)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _save(fig, out_dir, name):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    _plt().close(fig)
    return path.name


def _arms(results):
    return results.get("arms") or {}


def budget_series(results):
    """-> [(head_max_len, accuracy)] over the arms that vary only the budget.

    The A and B arms alone: every other arm changes the option count or the
    checkpoint as well, so putting them on a budget axis would attribute their
    difference to the budget."""
    rows = [a for a in _arms(results).values()
            if a["arm"] == "A-default" or a["arm"].startswith("B-")]
    return sorted((a["head_max_len"], round(a["accuracy"], 3)) for a in rows)


def raised_budgets_agree(results):
    """Whether every raised-budget arm landed on the same accuracy.

    If they did, the effect is a cliff the default sits below rather than a dial,
    and the caption says so instead of leaving the reader to infer a trend."""
    raised = {acc for budget, acc in budget_series(results) if budget != DEFAULT_BUDGET}
    return len(raised) == 1


def budget_cliff_plot(results, out_dir):
    series = budget_series(results)
    if len(series) < 2:
        raise NothingToPlot(
            "the budget panel needs the default arm and at least one B arm to "
            f"compare it against; found {len(series)}")

    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    budgets = [b for b, _ in series]
    accuracies = [a for _, a in series]

    ax.step(budgets, accuracies, where="post", color=SHADES[0], linewidth=1.6)
    ax.plot(budgets, accuracies, "o", color=SHADES[0], markersize=7)
    for budget, accuracy in series:
        ax.annotate(f"{accuracy:.3f}", (budget, accuracy), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=8)

    ax.axvline(DEFAULT_BUDGET, color=SHADES[2], linewidth=1.0, linestyle=":")
    ax.annotate("ships at 256", (DEFAULT_BUDGET, 0.06), textcoords="offset points",
                xytext=(7, 0), fontsize=8, color=SHADES[1])

    ax.set_xticks(budgets)
    ax.set_ylim(0, 1)
    _style(ax, "Raising the option budget buys one step and then nothing",
           "head_max_len (tokens shared across all 77 options)", "accuracy")
    if raised_budgets_agree(results):
        ax.text(0.5, -0.19, "384, 512 and 768 are identical to every decimal: a cliff, not a dial.",
                transform=ax.transAxes, ha="center", fontsize=8, color=SHADES[1])
    return _save(fig, out_dir, "budget-cliff")


def option_count_plot(results, out_dir):
    """Accuracy against option count with the budget untouched.

    This is the arm that separates the two explanations. If the budget were the
    problem, accuracy would be flat here, because nothing in this sweep changes
    how much room an option gets."""
    rows = sorted((a["n_options"], a["accuracy"], a["n_test"])
                  for a in _arms(results).values()
                  if a["arm"].startswith("K-") or a["arm"] == "A-default")
    if len([r for r in rows if r[0] != 77]) < 2:
        raise NothingToPlot(
            "the option-count panel needs the K arms; found "
            f"{len(rows)} arm(s) on the option axis")

    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    counts = [c for c, _, _ in rows]
    accuracies = [a for _, a, _ in rows]

    ax.plot(counts, accuracies, "-o", color=SHADES[0], linewidth=1.6, markersize=7)
    for count, accuracy, n_test in rows:
        ax.annotate(f"{accuracy:.3f}\nn={n_test}", (count, accuracy),
                    textcoords="offset points", xytext=(0, 10), ha="center", fontsize=8)

    ax.set_xticks(counts)
    ax.set_ylim(0, 1)
    _style(ax, "Accuracy falls as options are added, with the budget untouched",
           "options on the table", "accuracy")
    ax.text(0.5, -0.19,
            f"Every point runs at head_max_len={DEFAULT_BUDGET}. Nothing is truncated below 77 options.",
            transform=ax.transAxes, ha="center", fontsize=8, color=SHADES[1])
    return _save(fig, out_dir, "option-count")


def comparable_arms(results):
    """-> the arms that answer the same question, weakest first.

    Only the full 77-option arms. K-10 scores 0.743 against ten options and
    B-768 scores 0.488 against seventy-seven; on a single accuracy axis the
    shorter task reads as the better configuration, and an axis label does not
    undo that. The sweep gets `option-count` instead, where the option count is
    the variable rather than a footnote."""
    full = [a for a in _arms(results).values() if a["n_options"] == 77]
    return sorted(full, key=lambda a: a["accuracy"])


def arms_plot(results, out_dir):
    """Every 77-option arm against the two published numbers.

    The published values are drawn as reference lines and labelled as published,
    never as bars beside the measured ones, because they were produced elsewhere
    on a different label set."""
    rows = comparable_arms(results)
    if not rows:
        raise NothingToPlot(
            "no 77-option arms in the results file; this panel compares "
            "configurations at the full label set and the K sweep is not one")

    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = [a["arm"] for a in rows]
    accuracies = [a["accuracy"] for a in rows]

    ax.barh(labels, accuracies, color=SHADES[2], edgecolor=SHADES[0], linewidth=0.8)
    for i, accuracy in enumerate(accuracies):
        ax.annotate(f"{accuracy:.3f}", (accuracy, i), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=8)

    published = results.get("published_comparison") or {}
    for key, style in (("laya_banking77_accuracy", ":"), ("jev_banking77_accuracy", "--")):
        value = published.get(key)
        if value is None:
            continue
        ax.axvline(value, color=SHADES[0], linewidth=1.1, linestyle=style)
        ax.annotate(f"published {key.split('_')[0]} {value:.3f}", (value, len(rows) - 0.4),
                    textcoords="offset points", xytext=(4, 0), fontsize=8, rotation=90,
                    va="top")

    ax.set_xlim(0, 1)
    _style(ax, "What survives fixing the budget, all at 77 options", "accuracy", "")
    ax.text(0.5, -0.14,
            "Dashed lines are published numbers, not measured here; Jev was scored on 72 labels, Laya on 77.",
            transform=ax.transAxes, ha="center", fontsize=8, color=SHADES[1])
    return _save(fig, out_dir, "arms")


PLOTS = {
    "budget-cliff": budget_cliff_plot,
    "option-count": option_count_plot,
    "arms": arms_plot,
}


def write_all(results, out_dir):
    """-> {"written": [filenames], "skipped": {figure: reason}}.

    A figure that cannot be drawn honestly is skipped **with its reason
    reported**, and writes no file. The caller prints the skips; a silently
    missing chart and an empty one are both worse than a stated absence."""
    written, skipped = [], {}
    for name, plot in PLOTS.items():
        try:
            written.append(plot(results, out_dir))
        except NothingToPlot as e:
            skipped[name] = str(e)
    return {"written": written, "skipped": skipped}

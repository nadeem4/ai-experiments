"""Every figure in this experiment, drawn by code from the results file.

Nothing here is hand-drawn and nothing takes its own input: the charts and the
tables both read `typed_decisions/results/<tag>.json`, so a figure cannot drift
away from the numbers printed beside it. `report.py` calls `write_all` by
default, so the images are regenerated whenever the tables are, rather than being
something somebody has to remember.

**A figure with nothing to show raises.** A calibration panel for models that
return no probabilities, or cost-against-accuracy for a task with no ground
truth, would be an empty chart that a reader assumes means something. `write_all`
catches the refusal, writes no file, and reports the skip with its reason.

## Styling

Plain on purpose. No 3D, no gradients, no chart junk, no styling that stops
working at the size GitHub renders a PNG inline.

**Meaning is never encoded in colour alone.** Every series carries a marker shape
or a hatch as well as a shade of grey, and points are labelled directly rather
than through a legend where that fits, so the figures survive a grayscale print
and a colour-blind reader.

## What is drawn, and why each one earns its place

  * **cost-accuracy** -- one point per model per task. The winner-picking chart:
    the useful models are on the lower-right frontier, cheap and accurate. Points
    are labelled, so no legend lookup is needed.
  * **latency** -- p50 with a whisker to p95, grouped by task, because a p50-only
    bar chart hides a 10x tail and a tail is what hurts in production.
  * **position-bias** -- accuracy by gold position, grouped bars, one panel per
    task. This is the figure the protocol's prediction lives or dies on: flat
    bars for the decision models, sloped bars for the LLMs at 151 options.
  * **option-scaling** -- mean input tokens on the 4-option task against the
    151-option one. What the option list costs each model.
  * **calibration** -- a reliability diagram for the models that return a
    probability at all, against the diagonal.
"""
from pathlib import Path

DPI = 160
FIGSIZE_WIDE = (9.5, 5.2)
FIGSIZE_TALL = (9.5, 6.4)

# Shape first, shade second. A reader who cannot see the shades still has the
# markers and the hatches, and every point is labelled.
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*", "<", ">")
SHADES = ("#111111", "#444444", "#777777", "#999999", "#bbbbbb")
HATCHES = ("", "///", "...", "xxx", "\\\\\\")
GRID = {"color": "#cccccc", "linewidth": 0.6, "alpha": 0.9}

LOCAL_NOTE = ("local CPU, no network in it: not comparable to a hosted call")


class NothingToPlot(ValueError):
    """The metric this figure is about is absent from the results.

    Raised rather than drawing an empty chart. A published figure is read as a
    claim, and an empty one is a claim that nothing happened."""


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


def _tasks(summary):
    """Task names in the order they first appear, so panels do not reshuffle
    between runs."""
    return list(dict.fromkeys(row["task"] for row in summary.values()))


def _models(summary):
    return list(dict.fromkeys(row["model"] for row in summary.values()))


def _style_for(index):
    return (MARKERS[index % len(MARKERS)], SHADES[index % len(SHADES)],
            HATCHES[index % len(HATCHES)])


# --- the winner-picking chart -------------------------------------------------


def priced_out(summary):
    """Models that cost nothing, and so cannot go on a log cost axis.

    Laya runs on this machine, so its cost is a true zero. A log scale silently
    drops it, which would take the only free model off the frontier chart without
    a word -- the most misleading thing this figure could do. It is excluded
    explicitly and named in the caption instead."""
    return sorted({row["model"] for row in summary.values()
                   if not row.get("cost_per_1k_usd")})


def cost_accuracy_plot(results, out_dir):
    """Cost per 1,000 decisions against accuracy, one point per model per task.

    The lower-right corner is what you want: accurate and cheap. Points are
    labelled directly so the chart is readable without a legend, and the local
    model is drawn hollow because its zero cost is an artefact of running on this
    machine rather than a price anyone was quoted."""
    plt = _plt()
    summary = results.get("summary") or {}
    scored = [r for r in summary.values() if r.get("accuracy") is not None]
    if not scored:
        raise NothingToPlot("no row reports an accuracy: this task has no ground truth, "
                            "so there is no cost-against-accuracy frontier to draw")
    priced = [r for r in scored if r.get("cost_per_1k_usd")]
    if not priced:
        raise NothingToPlot("no row reports a non-zero cost per 1,000 decisions, and a "
                            "zero cannot go on the log axis this chart needs")
    free = priced_out(summary)

    tasks = _tasks(summary)
    fig, axes = plt.subplots(1, len(tasks), figsize=FIGSIZE_WIDE, squeeze=False)
    for panel, task in zip(axes[0], tasks):
        rows = [r for r in priced if r["task"] == task]
        for i, row in enumerate(sorted(rows, key=lambda r: r["model"])):
            marker, shade, _ = _style_for(i)
            panel.scatter(row["cost_per_1k_usd"], row["accuracy"], marker=marker, s=70,
                          facecolors="none" if row.get("local") else shade,
                          edgecolors=shade, linewidths=1.4, zorder=3)
            panel.annotate(row["model"], (row["cost_per_1k_usd"], row["accuracy"]),
                           textcoords="offset points", xytext=(7, 3), fontsize=8)
        n_options = rows[0]["n_options"] if rows else "?"
        _style(panel, f"{task} ({n_options} options)",
               "cost per 1,000 decisions (USD, log scale)", "accuracy")
        panel.set_xscale("log")
    fig.suptitle("Cost against accuracy: the useful models are lower and to the right",
                 fontsize=12, x=0.02, ha="left")
    caption = f"hollow marker = {LOCAL_NOTE}, and its zero cost is not a price"
    if free:
        caption = (f"Not shown, costs nothing to run so it has no place on a log cost "
                   f"axis: {', '.join(free)}. Its accuracy is in the table.")
    fig.text(0.02, -0.03, caption, fontsize=8, color="#555555")
    fig.tight_layout()
    return _save(fig, out_dir, "cost-accuracy")


# --- latency, with the tail visible ------------------------------------------


def latency_plot(results, out_dir):
    """p50 with a whisker up to p95, grouped by task.

    A p50-only bar chart hides a 10x tail, and a model with a 10x tail is worse
    in production than a slower steady one. The local model is hatched and
    labelled, because a CPU number and a network round trip are not a latency
    comparison."""
    plt = _plt()
    summary = results.get("summary") or {}
    timed = {k: r for k, r in summary.items() if r.get("p50_ms") is not None}
    if not timed:
        raise NothingToPlot("no row reports a latency: nothing in this store was timed")

    tasks = _tasks(timed)
    models = _models(timed)
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    width = 0.8 / max(len(tasks), 1)
    for t, task in enumerate(tasks):
        by_model = {r["model"]: r for r in timed.values() if r["task"] == task}
        xs, p50s, errs, hatches = [], [], [], []
        for m, model in enumerate(models):
            row = by_model.get(model)
            if row is None:
                continue
            xs.append(m + t * width - 0.4 + width / 2)
            p50s.append(row["p50_ms"])
            errs.append(max(0.0, (row.get("p95_ms") or row["p50_ms"]) - row["p50_ms"]))
            hatches.append(HATCHES[1] if row.get("local") else "")
        bars = ax.bar(xs, p50s, width=width * 0.9, color=SHADES[t % len(SHADES)],
                      edgecolor="#111111", linewidth=0.7, label=f"{task} (p50)",
                      yerr=[[0] * len(errs), errs], capsize=3,
                      error_kw={"ecolor": "#111111", "elinewidth": 0.9})
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=8, rotation=25, ha="right")
    ax.set_yscale("log")
    _style(ax, "Latency: bar is p50, whisker reaches p95",
           "", "milliseconds, successful attempt only (log scale)")
    ax.legend(fontsize=8, frameon=False)
    fig.text(0.02, -0.04, f"hatched bar = {LOCAL_NOTE}. Backoff is excluded from these "
                          f"numbers and is reported as wall clock in the table.",
             fontsize=8, color="#555555")
    fig.tight_layout()
    return _save(fig, out_dir, "latency")


# --- the figure the prediction lives or dies on -------------------------------


def position_bias_plot(results, out_dir):
    """Accuracy with the correct option placed first, middle and last.

    Grouped bars per model, one panel per task. Flat bars mean the model does not
    care where the answer sits; sloped bars mean it does, and the slope says which
    end it favours. The flip rate is annotated on each group, because how much a
    model moves and which positions it prefers are two different facts."""
    plt = _plt()
    bias = results.get("position_bias") or {}
    if not bias:
        raise NothingToPlot("no position-bias arm ran: there is no flip rate and no "
                            "gold-placement accuracy in this store")
    # A model that returned nothing usable under any ordering scores zero at every
    # position. That is a structural failure, not a position effect, so it is left
    # out and the exclusion is stated in the caption rather than drawn as three
    # flat bars at zero.
    excluded = sorted(k for k, v in bias.items() if v.get("n_valid") == 0)
    measured = {k: v for k, v in bias.items()
                if v.get("n_valid") != 0
                and any(v.get("gold", {}).get("accuracy", {}).get(p) is not None
                        for p in ("first", "middle", "last"))}
    if not measured:
        raise NothingToPlot("no model returned a usable answer under the gold-placement "
                            "arm, so accuracy by position cannot be drawn")

    tasks = list(dict.fromkeys(v["task"] for v in measured.values()))
    fig, axes = plt.subplots(1, len(tasks), figsize=FIGSIZE_TALL, squeeze=False, sharey=True)
    placements = ("first", "middle", "last")
    for panel, task in zip(axes[0], tasks):
        rows = sorted([v for v in measured.values() if v["task"] == task],
                      key=lambda v: v["model"])
        width = 0.8 / len(placements)
        for p, placement in enumerate(placements):
            xs = [i + p * width - 0.4 + width / 2 for i in range(len(rows))]
            ys = [(r["gold"]["accuracy"].get(placement) or 0.0) for r in rows]
            bars = panel.bar(xs, ys, width=width * 0.92, color=SHADES[p % len(SHADES)],
                             edgecolor="#111111", linewidth=0.7,
                             label=f"gold {placement}")
            for bar in bars:
                bar.set_hatch(HATCHES[p % len(HATCHES)])
        for i, r in enumerate(rows):
            rate = (r.get("flip") or {}).get("rate")
            if rate is not None:
                panel.annotate(f"flip {rate:.0%}", (i, 1.02), ha="center", fontsize=8,
                               color="#555555", annotation_clip=False)
        panel.set_xticks(range(len(rows)))
        panel.set_xticklabels([r["model"] for r in rows], fontsize=8, rotation=35, ha="right")
        panel.set_ylim(0, 1.12)
        n_examples = rows[0]["n_examples"] if rows else 0
        _style(panel, f"{task} ({rows[0]['n_options']} options, n={n_examples})",
               "", "accuracy")
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper center",
                      bbox_to_anchor=(1.05, -0.22), ncol=3)
    fig.suptitle("Position bias: same example, same options, only the order changes",
                 fontsize=12, x=0.02, ha="left")
    caption = ("Flat bars mean the model does not care where the answer sits. "
               "Any slope is pure error.")
    if excluded:
        caption += (f"\nLeft out, no usable answer under any ordering: "
                    f"{', '.join(excluded)}.")
    fig.text(0.02, -0.03, caption, fontsize=8, color="#555555")
    fig.tight_layout()
    return _save(fig, out_dir, "position-bias")


# --- what the option list costs ----------------------------------------------


def option_scaling_plot(results, out_dir):
    """Mean input tokens per call on the short option list against the long one.

    One line per model between the two tasks. A decision model should be close to
    flat; an LLM carries the whole option list in its prompt on every call."""
    plt = _plt()
    summary = results.get("summary") or {}
    tasks = _tasks(summary)
    if len(tasks) < 2:
        raise NothingToPlot("option-count scaling needs two tasks with different option "
                            f"counts; this store holds {len(tasks)}")
    with_tokens = {r["model"]: {} for r in summary.values()}
    for r in summary.values():
        if r.get("mean_input_tokens") is not None:
            with_tokens[r["model"]][r["task"]] = (r["mean_input_tokens"], r["n_options"])
    plotted = {m: v for m, v in with_tokens.items() if len(v) >= 2}
    if not plotted:
        raise NothingToPlot("no model reported input tokens on two tasks; a model that "
                            "reports no tokens is left out rather than drawn at zero")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    for i, (model, by_task) in enumerate(sorted(plotted.items())):
        marker, shade, _ = _style_for(i)
        xs = [by_task[t][1] for t in tasks if t in by_task]
        ys = [by_task[t][0] for t in tasks if t in by_task]
        ax.plot(xs, ys, marker=marker, color=shade, linewidth=1.4, markersize=7)
        ax.annotate(model, (xs[-1], ys[-1]), textcoords="offset points",
                    xytext=(8, 0), fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sorted({v[1] for by_task in plotted.values() for v in by_task.values()}))
    ax.get_xaxis().set_major_formatter(_plt().matplotlib.ticker.ScalarFormatter())
    _style(ax, "What the option list costs: mean input tokens per call",
           "options in the list (log scale)", "mean input tokens (log scale)")
    fig.text(0.02, -0.04, "A model whose line is flat is not paying for its options. "
                          "Models that report no token counts are absent rather than zero.",
             fontsize=8, color="#555555")
    fig.tight_layout()
    return _save(fig, out_dir, "option-scaling")


# --- calibration, for the models that return a probability at all -------------


def calibration_plot(results, out_dir):
    """A reliability diagram per probabilistic model, against the diagonal.

    Only the decision models appear: an LLM in structured mode returns a label
    and nothing else, so there is nothing to calibrate and an empty panel would
    read as a perfectly calibrated model."""
    plt = _plt()
    calibration = results.get("calibration") or {}
    drawable = {k: v for k, v in calibration.items() if (v.get("reliability") or {}).get("accuracy")}
    if not drawable:
        raise NothingToPlot("no model in this store returned probabilities, so there is "
                            "nothing to calibrate: an LLM in structured mode returns a "
                            "label and no distribution")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1.0,
            label="perfectly calibrated")
    for i, (key, row) in enumerate(sorted(drawable.items())):
        marker, shade, _ = _style_for(i)
        reliability = row["reliability"]
        # No temperature means the fit was refused for too few validation rows,
        # not that the model is uncalibrated. The raw curve is still the
        # measurement, and the legend says which it is.
        temperature = row.get("temperature")
        fitted = f"T={temperature:.2f}" if temperature is not None else "no T fitted"
        ax.plot(reliability["bin_centres"], reliability["accuracy"], marker=marker,
                color=shade, linewidth=1.4, markersize=6,
                label=f"{key} (ECE {row['ece_raw']:.3f}, {fitted})")
    _style(ax, "Calibration: confidence against how often the answer was right",
           "predicted probability of the chosen option", "share actually correct")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    n_bins = next(iter(drawable.values())).get("n_bins", "?")
    fig.text(0.02, -0.04, f"{n_bins} bins. The temperature was fitted on a validation split "
                          f"carved out of train, never on test.",
             fontsize=8, color="#555555")
    fig.tight_layout()
    return _save(fig, out_dir, "calibration")


PLOTS = {
    "cost-accuracy": cost_accuracy_plot,
    "latency": latency_plot,
    "position-bias": position_bias_plot,
    "option-scaling": option_scaling_plot,
    "calibration": calibration_plot,
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

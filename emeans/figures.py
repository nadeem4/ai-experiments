"""Every figure in this experiment, drawn from `results/<tag>/summary.json` alone.

Nothing is read from `runs/` and no number is typed here, so the committed
results rebuild the figures. A figure with nothing to show raises `NothingToPlot`;
`write_all` catches it, writes no file and records the reason, because a
published chart is read as a claim and an empty one claims nothing happened.

Greyscale only: series differ by marker and dash as well as shade, and are named
by a direct label rather than a legend. The constants are copied from `rerank`
(not imported) so each experiment stays self-contained.

  * **b-recall-over-time** -- recall@10 at nprobe 16 per month for the four arms,
    mean with the min-max range across seeds; the yearly arm's sawtooth shows.
  * **b-cumulative-maintenance** -- cumulative maintenance seconds per arm.
  * **b-end-state-gap** -- 2025 recall differences between arm pairs, paired
    bootstrap intervals, at nprobe 16 and 64.
  * **a-race** -- test MSE over Lloyd's 25-iteration MSE against epochs, with
    Lloyd's own curve over iterations; 1.0 is the match line.
  * **a-budget-recall** and **a-scanned-p95** -- recall@10 and p95 vectors
    scanned at nprobe 16 per wall-clock budget.
  * **c-dimension** -- epochs to match and speedup against embedding dimension.
"""
from pathlib import Path

DPI = 160
FIGSIZE = (9.0, 5.0)
SHADES = ("#111111", "#555555", "#888888", "#aaaaaa")
MARKERS = ("o", "s", "^", "D")
DASHES = ("-", "--", "-.", ":")
GRID = {"color": "#cccccc", "linewidth": 0.6, "alpha": 0.9}
NPROBE = "16"  # the nprobe the single-setting charts show


class NothingToPlot(ValueError):
    """The numbers this figure is about are absent."""


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(i):
    return {"color": SHADES[i % 4], "marker": MARKERS[i % 4], "linestyle": DASHES[i % 4]}


def _finish(fig, ax, out_dir, name, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(**GRID)
    fig.tight_layout()
    path = Path(out_dir) / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    _plt().close(fig)
    return str(path)


def _need(value, what):
    if not value:
        raise NothingToPlot(what)
    return value


def b_recall_over_time(s, out_dir):
    series = _need(s.get("b", {}).get("series"), "no part B records")
    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (arm, v) in enumerate(series.items()):
        x = range(len(v["months"]))
        r = v[f"recall{NPROBE}"]
        ax.fill_between(x, r["min"], r["max"], color=SHADES[i], alpha=0.15, linewidth=0)
        ax.plot(x, r["mean"], **_style(i), markersize=3, linewidth=1.2)
        ax.annotate(arm, (x[-1], r["mean"][-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                    va="center")
    months = next(iter(series.values()))["months"]
    ticks = [i for i, m in enumerate(months) if m.endswith("-01")] or [0]
    ax.set_xticks(ticks, [months[i] for i in ticks], rotation=45, ha="right")
    return _finish(fig, ax, out_dir, "b-recall-over-time", "Recall as the index grows (band: min-max over seeds)",
                   "month", f"recall@10 at nprobe {NPROBE}")


def b_cumulative_maintenance(s, out_dir):
    series = _need(s.get("b", {}).get("series"), "no part B records")
    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for i, (arm, v) in enumerate(series.items()):
        x = range(len(v["months"]))
        ax.plot(x, v["cumulative_s"], **_style(i), markersize=3, linewidth=1.2)
        ax.annotate(arm, (x[-1], v["cumulative_s"][-1]), xytext=(4, 0), textcoords="offset points", fontsize=8,
                    va="center")
    months = next(iter(series.values()))["months"]
    ticks = [i for i, m in enumerate(months) if m.endswith("-01")] or [0]
    ax.set_xticks(ticks, [months[i] for i in ticks], rotation=45, ha="right")
    return _finish(fig, ax, out_dir, "b-cumulative-maintenance",
                   "Cumulative maintenance CPU time (training + re-assignment)", "month", "seconds (cumulative)")


def b_end_state_gap(s, out_dir):
    rows = _need(s.get("b", {}).get("end_state"), "no 2025 records, so no end-state comparison")
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, sharey=True)
    labels = list(dict.fromkeys(f"{r['a']} - {r['b']}" for r in rows))
    for ax, p in zip(axes, (16, 64)):
        mine = {f"{r['a']} - {r['b']}": r for r in rows if r["nprobe"] == p}
        for y, label in enumerate(labels):
            r = mine[label]
            ax.errorbar(r["mean"], y, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", color="#111111",
                        capsize=3)
        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.set_title(f"nprobe {p}", fontsize=9)
        ax.set_xlabel("recall@10 difference (95% paired bootstrap)")
        ax.grid(**GRID)
    fig.suptitle("2025 end state: recall gap between arms", fontsize=10)
    fig.tight_layout()
    path = Path(out_dir) / "b-end-state-gap.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


def _panels(datasets):
    plt = _plt()
    fig, axes = plt.subplots(1, len(datasets), figsize=(FIGSIZE[0], FIGSIZE[1]), squeeze=False)
    return fig, axes[0]


def a_race(s, out_dir):
    a = _need(s.get("a"), "no part A records")
    fig, axes = _panels(a)
    for ax, (dataset, part) in zip(axes, a.items()):
        for i, (arm, curve) in enumerate(part["race"].items(), start=1):
            es = sorted(curve, key=int)
            ax.errorbar([int(e) for e in es], [curve[e]["mean"] for e in es],
                        yerr=[[curve[e]["mean"] - (curve[e]["lo"] or curve[e]["mean"]) for e in es],
                              [(curve[e]["hi"] or curve[e]["mean"]) - curve[e]["mean"] for e in es]],
                        label=arm, capsize=2, linewidth=1.2, markersize=4, **_style(i))
            ax.annotate(arm, (int(es[-1]), curve[es[-1]]["mean"]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, va="center")
        lloyd = part["lloyd_curve"]
        if lloyd:
            its = sorted(lloyd, key=int)
            ax.plot([int(i) for i in its], [lloyd[i] for i in its], **_style(0), linewidth=1.2, markersize=4)
            ax.annotate("lloyd (iterations)", (int(its[-1]), lloyd[its[-1]]), xytext=(4, 6),
                        textcoords="offset points", fontsize=8)
        ax.axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_xscale("log")
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("epochs (Lloyd: iterations)")
        ax.set_ylabel("test MSE / Lloyd 25-iteration MSE")
        ax.grid(**GRID)
    fig.suptitle("Part A: quality against training effort (below 1.0 matches Lloyd)", fontsize=10)
    fig.tight_layout()
    path = Path(out_dir) / "a-race.png"
    fig.savefig(path, dpi=DPI)
    _plt().close(fig)
    return str(path)


def _budget_metric(s, out_dir, name, field, title, ylabel):
    a = _need(s.get("a"), "no part A records")
    if not any(r["setting"].startswith("budget=") for part in a.values() for r in part["search"]):
        raise NothingToPlot("no budget units with search results (the pilot has none)")
    fig, axes = _panels(a)
    for ax, (dataset, part) in zip(axes, a.items()):
        rows = [r for r in part["search"] if r["setting"].startswith("budget=")]
        for i, arm in enumerate(sorted({r["arm"] for r in rows})):
            mine = sorted((r for r in rows if r["arm"] == arm), key=lambda r: int(r["setting"][7:-1]))
            xs = [int(r["setting"][7:-1]) for r in mine]
            ys = [r["search"][NPROBE][field] for r in mine]
            ax.plot(xs, ys, **_style(i), linewidth=1.2, markersize=4)
            ax.annotate(arm, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points", fontsize=8, va="center")
        ax.set_title(dataset, fontsize=9)
        ax.set_xlabel("training budget (s)")
        ax.set_ylabel(ylabel)
        ax.grid(**GRID)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(out_dir) / f"{name}.png"
    fig.savefig(path, dpi=DPI)
    _plt().close(fig)
    return str(path)


def a_budget_recall(s, out_dir):
    return _budget_metric(s, out_dir, "a-budget-recall", "recall10", f"Recall at a fixed training budget (nprobe {NPROBE})",
                          f"recall@10 at nprobe {NPROBE}")


def a_scanned_p95(s, out_dir):
    return _budget_metric(s, out_dir, "a-scanned-p95", "scanned_p95",
                          f"Search cost at a fixed training budget (nprobe {NPROBE})",
                          f"p95 vectors scanned per query at nprobe {NPROBE}")


def c_dimension(s, out_dir):
    c = _need(s.get("c"), "no part C records")
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    ds = sorted(c, key=int)
    xs = [int(d) for d in ds]
    matched = [(int(d), c[d]["epochs_to_match"]) for d in ds if c[d]["epochs_to_match"] is not None]
    if matched:
        axes[0].plot(*zip(*matched), **_style(0), linewidth=1.2)
    for d in ds:
        if c[d]["epochs_to_match"] is None:
            axes[0].annotate("no match\nin 5 epochs", (int(d), 5), ha="center", fontsize=7)
    axes[0].set_ylabel("epochs to match Lloyd")
    sp = [(int(d), c[d]["speedup"]) for d in ds if c[d]["speedup"] is not None]
    if sp:
        axes[1].plot(*zip(*sp), **_style(1), linewidth=1.2)
    axes[1].axhline(1.0, color="#555555", linewidth=0.8, linestyle="--")
    axes[1].set_ylabel("speedup over Lloyd (x)")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel("embedding dimension d")
        ax.grid(**GRID)
    fig.suptitle("Part C: does e-means' advantage change with dimension?", fontsize=10)
    fig.tight_layout()
    path = Path(out_dir) / "c-dimension.png"
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return str(path)


PLOTS = {"b-recall-over-time": b_recall_over_time, "b-cumulative-maintenance": b_cumulative_maintenance,
         "b-end-state-gap": b_end_state_gap, "a-race": a_race, "a-budget-recall": a_budget_recall,
         "a-scanned-p95": a_scanned_p95, "c-dimension": c_dimension}


def write_all(summary, out_dir):
    """-> {figure: path of the PNG, or "skipped: <reason>" with no file}."""
    out = {}
    for name, plot in PLOTS.items():
        try:
            out[name] = plot(summary, out_dir)
        except NothingToPlot as e:
            out[name] = f"skipped: {e}"
    return out

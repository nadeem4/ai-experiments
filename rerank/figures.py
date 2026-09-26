"""Every figure in this experiment, drawn by code from the report's own numbers.

Nothing here is hand-drawn and nothing takes its own input: the charts and the
CSVs beside them come out of the same pass, so a figure cannot drift away from
the table it sits next to. `evaluate.report` calls `write_all`, so the images are
regenerated whenever the numbers are.

**A figure with nothing to show raises.** `write_all` catches the refusal, writes
no file, and reports the skip with its reason. A published chart is read as a
claim, and an empty one claims that nothing happened.

## Styling

Plain, and greyscale. Meaning is never carried by colour alone: series are told
apart by marker and hatch as well as shade, and values are labelled directly
rather than through a legend, so the figures survive a grayscale print.

## What is drawn, and why each earns its place

  * **per-query-delta** -- every query's nDCG@10 change against the BM25 floor,
    sorted, with the queries that could not move counted separately. This is the
    experiment's central claim in one picture: a mean is a summary of this shape.
  * **score-distribution** -- how many distinct values each scorer actually used.
    A model answering on a coarse grid leaves passages tied, and a tie keeps
    BM25's order, so this is what says how much of the ranking is really the
    model's.
  * **latency** -- p50 with a whisker to p95, network and CPU grouped apart
    because they are not the same measurement.
  * **win-loss** -- how many queries each method improved, hurt and left alone.
"""
from pathlib import Path

DPI = 160
FIGSIZE = (9.0, 5.0)

SHADES = ("#111111", "#555555", "#888888", "#aaaaaa")
HATCHES = ("", "///", "...", "xxx")
GRID = {"color": "#cccccc", "linewidth": 0.6, "alpha": 0.9}

# Measured over the network rather than on this machine, so its latency is a
# round trip and does not belong on the same axis as a local forward pass.
HOSTED = ("jev-score", "jev-noul")


class NothingToPlot(ValueError):
    """The numbers this figure is about are absent.

    Raised rather than drawing an empty panel, which a reader assumes means a
    measurement came out flat."""


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


def _reranked(results):
    """The methods measured against the floor. BM25 is the floor, not a rival."""
    return [s for s in results["methods"] if s.get("ndcg@10_vs_bm25")]


def split_by_movability(deltas, relevant):
    """-> (queries that could move, queries that could not).

    A query whose candidates hold nothing the qrels judge relevant scores zero
    for every method whatever order it picks. Those are not a model declining to
    act, and plotting them among the rest turns a structural zero into an
    apparent decision."""
    movable = {q: d for q, d in deltas.items() if relevant.get(q, 0) > 0}
    stuck = sorted(q for q in deltas if relevant.get(q, 0) == 0)
    return movable, stuck


def per_query_delta_plot(results, deltas, relevant, out_dir):
    if not deltas:
        raise NothingToPlot("no per-query differences were computed")

    plt = _plt()
    methods = sorted(deltas)
    fig, axes = plt.subplots(1, len(methods), figsize=(4.6 * len(methods), 4.4),
                             sharey=True, squeeze=False)
    for index, (ax, method) in enumerate(zip(axes[0], methods)):
        movable, stuck = split_by_movability(deltas[method], relevant)
        values = sorted(movable.values())
        ax.bar(range(len(values)), values, width=1.0,
               color=SHADES[1], edgecolor="none")
        ax.axhline(0, color=SHADES[0], linewidth=1.0)
        gained = sum(1 for v in values if v > 0)
        lost = sum(1 for v in values if v < 0)
        _style(ax, method, f"{len(values)} queries that could move, sorted",
               "nDCG@10 against BM25" if index == 0 else "")
        ax.text(0.02, 0.96, f"better on {gained}\nworse on {lost}\n"
                            f"{len(stuck)} could not move at all",
                transform=ax.transAxes, va="top", fontsize=8, color=SHADES[1])
    return _save(fig, out_dir, "per-query-delta")


def score_distribution_plot(results, records, out_dir):
    """How many distinct values each scorer used, against how many calls it made."""
    counts = {}
    for record in records:
        if record["score"] is None:
            continue
        counts.setdefault(record["method"], set()).add(record["score"])
    if not counts:
        raise NothingToPlot("no scored calls in the run")

    calls = {}
    for record in records:
        if record["score"] is not None:
            calls[record["method"]] = calls.get(record["method"], 0) + 1

    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    methods = sorted(counts)
    distinct = [len(counts[m]) for m in methods]
    ax.barh(methods, distinct, color=SHADES[2], edgecolor=SHADES[0], linewidth=0.8)
    for i, method in enumerate(methods):
        ax.annotate(f"{len(counts[method]):,} distinct of {calls[method]:,} calls",
                    (len(counts[method]), i), textcoords="offset points",
                    xytext=(4, 0), va="center", fontsize=8)
    ax.set_xscale("log")
    _style(ax, "How many different answers each scorer actually gave",
           "distinct score values (log)", "")
    ax.text(0.5, -0.16,
            "Fewer distinct values means more tied passages, and a tie keeps BM25's order.",
            transform=ax.transAxes, ha="center", fontsize=8, color=SHADES[1])
    return _save(fig, out_dir, "score-distribution")


def latency_plot(results, out_dir):
    rows = [s for s in results["methods"]
            if (s.get("latency") or {}).get("p50_ms") is not None]
    if not rows:
        raise NothingToPlot("no method recorded a latency")

    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    rows.sort(key=lambda s: s["latency"]["p50_ms"])
    names = [s["method"] for s in rows]
    p50 = [s["latency"]["p50_ms"] for s in rows]
    p95 = [s["latency"]["p95_ms"] for s in rows]
    for i, s in enumerate(rows):
        hosted = s["method"] in HOSTED
        ax.barh(i, p50[i], color=SHADES[2] if hosted else SHADES[3],
                hatch="///" if hosted else "", edgecolor=SHADES[0], linewidth=0.8)
        ax.plot([p50[i], p95[i]], [i, i], color=SHADES[0], linewidth=1.2)
        ax.annotate(f"{p50[i]:,.0f} / {p95[i]:,.0f} ms", (p95[i], i),
                    textcoords="offset points", xytext=(5, 0), va="center", fontsize=8)
    ax.set_yticks(range(len(names)), names)
    ax.set_xscale("log")
    _style(ax, "Latency per call: bar is p50, whisker reaches p95", "milliseconds (log)", "")
    ax.text(0.5, -0.16,
            "Hatched is a network round trip; plain is this machine's CPU. Not the same measurement.",
            transform=ax.transAxes, ha="center", fontsize=8, color=SHADES[1])
    return _save(fig, out_dir, "latency")


def win_loss_plot(results, out_dir):
    rows = _reranked(results)
    if not rows:
        raise NothingToPlot("no method was measured against the BM25 floor")

    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = [s["method"] for s in rows]
    parts = [("better", SHADES[1], ""), ("same", SHADES[3], "..."), ("worse", SHADES[2], "///")]
    left = [0] * len(rows)
    for label, shade, hatch in parts:
        widths = [s["ndcg@10_vs_bm25"][label] for s in rows]
        ax.barh(names, widths, left=left, color=shade, hatch=hatch,
                edgecolor=SHADES[0], linewidth=0.8, label=label)
        for i, (w, l) in enumerate(zip(widths, left)):
            if w:
                ax.annotate(str(w), (l + w / 2, i), ha="center", va="center", fontsize=8)
        left = [l + w for l, w in zip(left, widths)]
    ax.legend(fontsize=8, frameon=False, ncols=3, loc="lower right")
    _style(ax, "Queries improved, unchanged and made worse", "queries", "")
    return _save(fig, out_dir, "win-loss")


def write_all(results, deltas, relevant, records, out_dir):
    """-> {"written": [filenames], "skipped": {figure: reason}}.

    A figure that cannot be drawn honestly is skipped **with its reason**, and
    writes no file. A silently missing chart and an empty one are both worse than
    a stated absence."""
    plots = {
        "per-query-delta": lambda: per_query_delta_plot(results, deltas, relevant, out_dir),
        "score-distribution": lambda: score_distribution_plot(results, records, out_dir),
        "latency": lambda: latency_plot(results, out_dir),
        "win-loss": lambda: win_loss_plot(results, out_dir),
    }
    written, skipped = [], {}
    for name, plot in plots.items():
        try:
            written.append(plot())
        except NothingToPlot as e:
            skipped[name] = str(e)
    return {"written": written, "skipped": skipped}

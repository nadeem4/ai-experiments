"""Write the lab site's typed_decisions data out of `results/full.json`.

    python -m typed_decisions.scripts.export_site_data

`typed_decisions.report` calls this as its last step, right after `figures`, for
the same reason: a chart or a number that is produced anywhere other than the
results file drifts away from it the first time the run is redone, and nobody
notices because prose is not diffed. This reads the results file and nothing
else, and copies the committed figures into the site's public directory so the
page shows the charts the report drew rather than redrawing them in a browser.

Only what the page renders goes out. The comparisons the page makes -- how far
each model's median moves from 4 options to 151, what share of an LLM's prompt is
the option list -- are computed here so they can be tested.
"""
import argparse
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "typed_decisions" / "results" / "full.json"
FIGURES = ROOT / "typed_decisions" / "results" / "figures"
SITE_DATA = ROOT / "site" / "data" / "typed-decisions.json"
PUBLIC = ROOT / "site" / "public"

# The two figures the page shows, and the names they are published under.
SHOWN_FIGURES = {"latency": "latency.png", "position_bias": "position-bias.png"}
PREFIX = "typed-decisions-"


def _model_rows(summary):
    """One row per model and task, fastest first within a task, locals last.

    A locally measured median is not comparable to a hosted round trip, so a
    local row never sorts into the speed ranking even when its number is lower.
    A row that never returned an answer keeps its nulls: a structural refusal is
    not a latency of zero and not an accuracy of zero."""
    rows = [
        {
            "model": s["model"],
            "model_id": s["model_id"],
            "task": s["task"],
            "n_options": s["n_options"],
            "transport": s["transport"],
            "local": s["local"],
            "n": s["n"],
            "p50_ms": s["p50_ms"],
            "p95_ms": s["p95_ms"],
            "tail_ratio": s["tail_ratio"],
            "accuracy": s["accuracy"],
            "accuracy_ci": s["accuracy_ci"],
            "valid_rate": s["valid_rate"],
            "validity": s["validity"],
            "retries": s["retries"],
            "cost_per_1k_usd": s["cost_per_1k_usd"],
            "cost_per_correct_usd": s["cost_per_correct_usd"],
            "mean_input_tokens": s["mean_input_tokens"],
            "mean_output_tokens": s["mean_output_tokens"],
            "returns_probability": s["returns_probability"],
        }
        for s in summary.values()
    ]
    return sorted(rows, key=lambda r: (r["n_options"], r["local"], r["p50_ms"] or float("inf")))


def _latency_rise(rows):
    """What each model's median costs when the answer space goes from 4 to 151.

    A model that could not answer one of the two tasks has no rise to report
    rather than a rise computed against a missing number."""
    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["task"]] = row["p50_ms"]
    rise = [
        {"model": model, "small_p50_ms": tasks["ag_news"], "large_p50_ms": tasks["clinc150"],
         "rise": tasks["clinc150"] / tasks["ag_news"] - 1}
        for model, tasks in by_model.items()
        if tasks.get("ag_news") and tasks.get("clinc150")
    ]
    return sorted(rise, key=lambda r: r["rise"])


def _position_bias(bias):
    return [
        {
            "model": b["model"],
            "task": b["task"],
            "n_options": b["n_options"],
            "n_examples": b["n_examples"],
            "n_calls": b["n_calls"],
            "n_valid": b["n_valid"],
            "flip_rate": b["flip"]["rate"],
            "gold_first": b["gold"]["accuracy"]["first"],
            "gold_middle": b["gold"]["accuracy"]["middle"],
            "gold_last": b["gold"]["accuracy"]["last"],
            "spread": b["gold"]["spread"],
            "mean_position": b["positions"]["mean_normalised"],
        }
        for b in bias.values()
    ]


def _option_share(overhead, tasks):
    """What share of a billed prompt is the option list, per task.

    A provider that reported fewer prompt tokens with the option list than
    without it has not measured the options, so it is named and left out rather
    than folded into a range it would widen."""
    out = []
    for task, meta in tasks.items():
        shares, unusable = {}, []
        for key, o in overhead.items():
            if not key.endswith(f"/{task}"):
                continue
            tokens, prompt = o.get("option_tokens"), o.get("prompt_tokens_with_options")
            if tokens is None or prompt is None:
                continue  # no options-free probe for this model; nothing to divide
            if tokens < 0:
                unusable.append(key)
            else:
                shares[key.split("/")[0]] = tokens / prompt
        out.append({
            "task": task,
            "n_options": meta["n_options"],
            "n_models": len(shares),
            "min_share": min(shares.values()) if shares else None,
            "max_share": max(shares.values()) if shares else None,
            "unusable": sorted(unusable),
        })
    return out


def _probability(rows):
    """Which models hand back a distribution, read off the records.

    A model counts as returning one if any of its rows did: a task it refused
    outright says nothing about the capability."""
    returns = {r["model"] for r in rows if r["returns_probability"]}
    label_only = {r["model"] for r in rows} - returns
    return {"returns": sorted(returns), "label_only": sorted(label_only)}


def png_size(path):
    """Width and height out of a PNG's IHDR, so the page can reserve the box."""
    header = path.read_bytes()[16:24]
    width, height = struct.unpack(">II", header)
    return width, height


def copy_figures(source=FIGURES, public=PUBLIC):
    """The charts the report drew, copied as they are. A figure the report did
    not draw is absent from the page rather than a broken image on it."""
    public.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, filename in SHOWN_FIGURES.items():
        if not (source / filename).exists():
            continue
        published = PREFIX + filename
        target = public / published
        target.write_bytes((source / filename).read_bytes())
        width, height = png_size(target)
        out[name] = {"src": f"/{published}", "width": width, "height": height}
    return out


def build(results, figures=None):
    rows = _model_rows(results["summary"])
    return {
        "tag": results["tag"],
        "written_at": results["written_at"],
        "n_calls": results["n_calls"],
        "spend_usd": results["total_recorded_spend_usd"],
        "tasks": [{"task": task, **meta} for task, meta in results["tasks"].items()],
        "models": rows,
        "latency_rise": _latency_rise(rows),
        "position_bias": _position_bias(results["position_bias"]),
        "option_share": _option_share(results["option_overhead"], results["tasks"]),
        "calibration": [
            {"model": c["model"], "task": c["task"], "temperature": c["temperature"],
             "ece_raw": c["ece_raw"], "ece_scaled": c["ece_scaled"],
             "n_test": c["n_test"], "n_validation": c["n_validation"]}
            for c in results["calibration"].values()
        ],
        "probability": _probability(rows),
        "laya_cfg": results["model_resolution"]["laya"]["cfg"],
        "figures": figures or {},
    }


def write(results, path=SITE_DATA, source=FIGURES, public=PUBLIC):
    data = build(results, copy_figures(source, public))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=RESULTS)
    args = p.parse_args(argv)
    if not args.results.exists():
        raise SystemExit(f"no results file at {args.results}; run "
                         f"`python -m typed_decisions.report --tag full` first")
    print(f"wrote {write(json.loads(args.results.read_text(encoding='utf-8')))}")


if __name__ == "__main__":
    main()

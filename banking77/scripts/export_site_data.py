"""Write the lab site's banking77 data out of `results/full/summary.json`.

    python -m banking77.scripts.export_site_data

`banking77.report` calls this as its last step, the same way the typed_decisions
report calls its figures, so the page cannot drift from the numbers the report
just printed. This reads the results file and nothing else: a headline typed into
a web page by hand is correct until the sweep is next run, and nobody diffs a
paragraph.

Only the fields the page renders go out, and every comparison the page makes --
what the budget buys, what it leaves on the table, what the workaround cost -- is
computed here rather than in the page, so the arithmetic is testable.
"""
import argparse
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "banking77" / "results" / "full" / "summary.json"
FIGURES = ROOT / "banking77" / "results" / "full" / "figures"
SITE_DATA = ROOT / "site" / "data" / "banking77.json"
PUBLIC = ROOT / "site" / "public"

# The figures the page shows, and the names they are published under.
SHOWN_FIGURES = {"arms": "arms.png", "budget_cliff": "budget-cliff.png",
                 "option_count": "option-count.png"}
PREFIX = "banking77-"

DEFAULT_ARM = "A-default"
RAISED_ARM = "B-384"
WORKAROUND_ARM = "C-coarse-fine"
BUDGET_ARMS = ("B-384", "B-512", "B-768")


def _arm(report):
    return {
        "arm": report["arm"],
        "n_test": report["n_test"],
        "n_options": report["n_options"],
        "head_max_len": report["head_max_len"],
        "checkpoint": report["checkpoint"],
        "accuracy": report["accuracy"],
        "ci95": report["accuracy_ci95"],
        "top5": report["top5_accuracy"],
        "macro_f1": report["macro_f1"],
        "ece_raw": report["ece_raw"],
        "temperature": report.get("temperature"),
        "ece_after_temperature": report.get("ece_after_temperature"),
        "p50_ms": report["latency_ms_p50"],
        "p95_ms": report["latency_ms_p95"],
    }


def _truncation(diagnostics):
    """One row per arm that ran a single flat question.

    Arm C's diagnostics are one block per step and carry no top-level option
    count, so it has no row here rather than a row of blanks."""
    return [
        {
            "arm": arm,
            "head_max_len": d["head_max_len"],
            "n_options": d["n_options"],
            "mean_text_tokens_per_option": d["mean_text_tokens_per_option"],
            "truncated_fraction": d["truncated_fraction"],
            "truncated_options": d["truncated_options"],
            "collided_options": d["collided_options"],
            "identical_pairs": d["identical_pairs"],
        }
        for arm, d in diagnostics.items()
        if "n_options" in d
    ]


def _headline(arms, published):
    """What the budget bought, against what the published comparison asks for."""
    default = arms[DEFAULT_ARM]["accuracy"]
    raised = arms[RAISED_ARM]["accuracy"]
    workaround = arms[WORKAROUND_ARM]["accuracy"]
    deficit = (published - default) * 100
    gain = (raised - default) * 100
    identical = [name for name in BUDGET_ARMS
                 if name in arms and arms[name]["accuracy"] == raised]
    return {
        "default_arm": DEFAULT_ARM,
        "raised_arm": RAISED_ARM,
        "workaround_arm": WORKAROUND_ARM,
        "deficit_points": deficit,
        "budget_gain_points": gain,
        "share_of_deficit": gain / deficit,
        "remaining_points": (published - raised) * 100,
        "workaround_vs_default_points": (workaround - default) * 100,
        "workaround_vs_raised_points": (workaround - raised) * 100,
        "identical_budget_arms": identical,
        "saturates_at": min(arms[name]["head_max_len"] for name in identical),
    }


def png_size(path):
    """Width and height out of a PNG's IHDR, so the page can reserve the box."""
    width, height = struct.unpack(">II", path.read_bytes()[16:24])
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
    arms = results["arms"]
    published = results["published_comparison"]
    diagnostics = results["truncation_diagnostics"]
    return {
        "tag": results["tag"],
        "device": arms[DEFAULT_ARM]["latency_device"],
        "published": {
            "source": published["source"],
            "laya": published["laya_banking77_accuracy"],
            "jev": published["jev_banking77_accuracy"],
            "note": published["note"],
        },
        "arms": [_arm(a) for a in arms.values()],
        "option_sweep": results["option_count_sweep"],
        "mcnemar": results["mcnemar"],
        "truncation": _truncation(diagnostics),
        "collisions": diagnostics[DEFAULT_ARM]["collisions"],
        "headline": _headline(arms, published["jev_banking77_accuracy"]),
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
                         f"`python -m banking77.report --tag full` first")
    print(f"wrote {write(json.loads(args.results.read_text(encoding='utf-8')))}")


if __name__ == "__main__":
    main()

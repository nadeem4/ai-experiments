"""Turns `decisions.jsonl` into the reported numbers.

    python -m banking77.report --tag pilot

Everything reported is measured on the test split. The one fitted quantity is the
temperature, and it is fitted on the validation split carved out of train.
"""
import argparse
import json
import statistics
from pathlib import Path

from banking77 import arms as arms_mod
from banking77 import metrics, options, store

EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_DIR = EXPERIMENT_DIR / "runs"
RESULTS_DIR = EXPERIMENT_DIR / "results"


def _rows(records, arm, split):
    return [r for r in records if r["arm"] == arm and r.get("split") == split]


def _vectors(rows, labels):
    """-> (predictions, gold, probability dicts, logit rows, gold indices)."""
    pred = [r["prediction"] for r in rows]
    gold = [r["gold"] for r in rows]
    probs = [r["probabilities"] for r in rows]
    logit_rows = [metrics.to_logits([p.get(label, 0.0) for label in labels]) for p in probs]
    gold_idx = [labels.index(g) for g in gold]
    return pred, gold, probs, logit_rows, gold_idx


def arm_report(records, arm, seed):
    test = _rows(records, arm, "test")
    val = _rows(records, arm, "val")
    if not test:
        return None
    labels = sorted({r["gold"] for r in test} | set(test[0]["probabilities"]))
    pred, gold, probs, logit_rows, gold_idx = _vectors(test, labels)
    correct = metrics.correctness(pred, gold)
    low, high = metrics.bootstrap_ci(correct, n=1000, seed=seed)
    conf = [max(p.values()) for p in probs]

    out = {
        "arm": arm,
        "n_test": len(test),
        "n_options": test[0]["n_options"],
        "head_max_len": test[0]["head_max_len"],
        "max_len": test[0]["max_len"],
        "checkpoint": test[0]["checkpoint"] or "english (repo root)",
        "accuracy": metrics.accuracy(pred, gold),
        "accuracy_ci95": [low, high],
        "top5_accuracy": metrics.top_k_accuracy(probs, gold, k=5),
        "macro_f1": metrics.macro_f1(pred, gold, labels),
        "ece_raw": metrics.ece(conf, correct, bins=15),
        "latency_ms_p50": statistics.quantiles([r["latency_ms"] for r in test], n=100)[49],
        "latency_ms_p95": statistics.quantiles([r["latency_ms"] for r in test], n=100)[94],
        "latency_device": "cpu",
    }
    if val:
        _, _, _, val_logits, val_gold_idx = _vectors(val, labels)
        t = metrics.fit_temperature(val_logits, val_gold_idx)
        scaled = [metrics.apply_temperature(z, t) for z in logit_rows]
        out["temperature"] = t
        out["n_val"] = len(val)
        out["ece_after_temperature"] = metrics.ece([max(p) for p in scaled], correct, bins=15)
        out["accuracy_after_temperature"] = metrics.accuracy(
            [labels[max(range(len(p)), key=p.__getitem__)] for p in scaled], gold
        )
    return out


def paired(records, a, b):
    """McNemar over the examples both arms actually ran."""
    rows_a = {r["example_id"]: r for r in _rows(records, a, "test")}
    rows_b = {r["example_id"]: r for r in _rows(records, b, "test")}
    shared = sorted(set(rows_a) & set(rows_b))
    if not shared:
        return None
    ca = [int(rows_a[i]["prediction"] == rows_a[i]["gold"]) for i in shared]
    cb = [int(rows_b[i]["prediction"] == rows_b[i]["gold"]) for i in shared]
    return {
        "a": a, "b": b, "n_paired": len(shared),
        "accuracy_a": sum(ca) / len(ca), "accuracy_b": sum(cb) / len(cb),
        **metrics.mcnemar(ca, cb),
    }


def build(tag, seed, out=None):
    run_dir = (Path(out) / "runs" if out else RUN_DIR) / tag
    records = store.load(run_dir / "decisions.jsonl")
    present = [name for name in arms_mod.ARMS if any(r["arm"] == name for r in records)]
    reports = {name: arm_report(records, name, seed) for name in present}
    reports = {k: v for k, v in reports.items() if v}

    diagnostics = {}
    for path in sorted((run_dir / "diagnostics").glob("*.json")) if (run_dir / "diagnostics").exists() else []:
        d = json.loads(path.read_text(encoding="utf-8"))
        if "n_options" in d:
            diagnostics[path.stem] = {k: v for k, v in d.items() if k != "options_as_the_model_sees_them"}
        else:  # coarse-fine: one block per step
            diagnostics[path.stem] = {
                "coarse": {k: v for k, v in d["coarse"].items() if k != "options_as_the_model_sees_them"},
                "fine_worst_group": max(
                    ({"group": g, **{k: v for k, v in s.items() if k != "options_as_the_model_sees_them"}}
                     for g, s in d["fine"].items()),
                    key=lambda s: s["identical_pairs"],
                ),
            }

    comparisons = [p for p in (
        [paired(records, "A-default", b) for b in ("B-384", "B-512", "B-768", "E-english", "C-coarse-fine")]
    ) if p]

    option_sweep = [
        {
            "arm": name,
            "n_options": reports[name]["n_options"],
            "tokens_per_option": diagnostics.get(name, {}).get("mean_text_tokens_per_option"),
            "identical_pairs": diagnostics.get(name, {}).get("identical_pairs"),
            "accuracy": reports[name]["accuracy"],
            "accuracy_ci95": reports[name]["accuracy_ci95"],
        }
        for name in arms_mod.OPTION_SWEEP if name in reports
    ]

    return {
        "tag": tag,
        "published_comparison": {
            "source": "convaiinnovations/laya model card",
            "laya_banking77_accuracy": 0.425,
            "jev_banking77_accuracy": 0.870,
            "note": "published, not measured here; Jev was scored on 72 labels, Laya on 77",
        },
        "arms": reports,
        "truncation_diagnostics": diagnostics,
        "option_count_sweep": option_sweep,
        "mcnemar": comparisons,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="pilot")
    p.add_argument("--seed", type=int, default=arms_mod.SEED)
    p.add_argument("--out", default=str(EXPERIMENT_DIR),
                   help="write runs/ and results/ under here (default: the experiment)")
    args = p.parse_args(argv)

    report = build(args.tag, args.seed, Path(args.out))
    results_dir = Path(args.out) / "results" / args.tag
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "summary.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'arm':16} {'n':>5} {'opts':>5} {'hml':>5} {'acc':>7} {'ci95':>16} {'top5':>7} "
          f"{'mF1':>7} {'ECE':>6} {'ECE_T':>6} {'p50ms':>7}")
    for name, r in report["arms"].items():
        ci = f"[{r['accuracy_ci95'][0]:.3f},{r['accuracy_ci95'][1]:.3f}]"
        print(f"{name:16} {r['n_test']:5d} {r['n_options']:5d} {r['head_max_len']:5d} "
              f"{r['accuracy']:7.3f} {ci:>16} {r['top5_accuracy']:7.3f} {r['macro_f1']:7.3f} "
              f"{r['ece_raw']:6.3f} {r.get('ece_after_temperature', float('nan')):6.3f} "
              f"{r['latency_ms_p50']:7.1f}")
    for c in report["mcnemar"]:
        print(f"McNemar {c['a']} vs {c['b']}: n10={c['n10']} n01={c['n01']} p={c['p_value']:.3g}")
    print(f"\nwrote {out}")

    # Always, so a chart cannot drift from the table it was printed beside.
    from . import figures

    figure_dir = results_dir / "figures"
    drawn = figures.write_all(report, figure_dir)
    for written in drawn["written"]:
        print(f"  figure: {figure_dir / written}")
    for name, reason in drawn["skipped"].items():
        print(f"  figure {name} SKIPPED: {reason}")

    # Same reason as the figures: the lab site reads a generated file, so no
    # number on the page can drift from the table above. The pilot writes beside
    # the full run's results and must not export over its data.
    if args.tag == "full":
        from .scripts import export_site_data

        print(f"  site data: {export_site_data.write(report, source=figure_dir)}")


if __name__ == "__main__":
    main()

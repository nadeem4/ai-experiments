"""Reads the call log and prints the tables.

    python -m latency.report --tag pilot

Laya is printed in its own group and labelled, because it ran on this machine's
CPU with no network in it: its latency is not comparable to a hosted call.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import stats, store

RUN_DIR = Path("latency/runs")
LOCAL = {"laya"}  # not comparable to a hosted call; grouped separately everywhere


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def summarise(records):
    """-> {"<model>/<task>": row}. Nothing is dropped before scoring: an answer that
    could not be parsed is a wrong answer, not a missing one."""
    groups = defaultdict(list)
    for r in records:
        groups[f"{r['model']}/{r['task']}"].append(r)

    out = {}
    for name, rows in groups.items():
        timed = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        walls = [r["wall_ms"] for r in rows if r["wall_ms"] is not None]
        validity = defaultdict(int)
        for r in rows:
            validity[r["validity"]] += 1
        costs = [r["market_cost"] for r in rows if r["market_cost"] is not None]
        correct = [r["correct"] for r in rows if r["correct"] is not None]
        tokens_in = [r["input_tokens"] for r in rows if r["input_tokens"] is not None]
        out[name] = {
            "model": rows[0]["model"],
            "model_id": rows[0]["model_id"],
            "task": rows[0]["task"],
            "n_options": rows[0]["n_options"],
            "structured": rows[0]["structured"],
            "local": rows[0]["model"] in LOCAL,
            "n": len(rows),
            "n_timed": len(timed),
            "p50_ms": stats.percentile(timed, 50),
            "p95_ms": stats.percentile(timed, 95),
            "wall_p50_ms": stats.percentile(walls, 50),
            "wall_p95_ms": stats.percentile(walls, 95),
            "mean_input_tokens": _mean(tokens_in),
            "mean_output_tokens": _mean([r["output_tokens"] for r in rows]),
            "n_tokens_reported": len(tokens_in),
            "cost_usd": sum(costs) if costs else None,
            "cost_per_1k_usd": (sum(costs) / len(rows) * 1000) if costs else None,
            "validity": dict(validity),
            "valid": validity["valid"],
            "valid_rate": validity["valid"] / len(rows),
            "accuracy": (sum(correct) / len(correct)) if correct else None,
            "accuracy_ci": stats.bootstrap_ci(correct) if correct else None,
            "retries": sum(r["retries"] for r in rows),
            "failed": sum(1 for r in rows if r["failed"]),
        }
    return out


def choices(records, model, task):
    return {r["example_id"]: r["choice"] for r in records if r["model"] == model and r["task"] == task}


def _fmt(value, spec="{:.0f}"):
    return "n/a" if value is None else spec.format(value)


def _rows_table(summary, keys):
    print(f"{'model':<10} {'model id':<30} {'n':>4} {'p50 ms':>8} {'p95 ms':>8} {'wall p50':>9} "
          f"{'in tok':>7} {'out tok':>8} {'$/1k':>9} {'valid':>7} {'acc':>16} {'rtry':>5}")
    for key in keys:
        s = summary[key]
        acc = "n/a (no gold)" if s["accuracy"] is None else \
            f"{s['accuracy']:.3f} [{s['accuracy_ci'][0]:.2f},{s['accuracy_ci'][1]:.2f}]"
        print(f"{s['model']:<10} {s['model_id'][:30]:<30} {s['n']:>4} {_fmt(s['p50_ms']):>8} "
              f"{_fmt(s['p95_ms']):>8} {_fmt(s['wall_p50_ms']):>9} {_fmt(s['mean_input_tokens']):>7} "
              f"{_fmt(s['mean_output_tokens'], '{:.1f}'):>8} {_fmt(s['cost_per_1k_usd'], '{:.4f}'):>9} "
              f"{s['valid_rate']:>6.0%} {acc:>16} {s['retries']:>5}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="pilot")
    args = p.parse_args(argv)

    run_dir = RUN_DIR / args.tag
    records = store.load(run_dir / "calls.jsonl")
    if not records:
        raise SystemExit(f"no calls recorded under {run_dir}")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    summary = summarise(records)

    for task in dict.fromkeys(r["task"] for r in records):
        keys = [k for k, s in summary.items() if s["task"] == task]
        hosted = [k for k in keys if not summary[k]["local"]]
        local = [k for k in keys if summary[k]["local"]]
        n_options = summary[keys[0]]["n_options"]
        print(f"\n=== {task} ({n_options} options) -- hosted, via the Vercel AI Gateway ===")
        _rows_table(summary, sorted(hosted, key=lambda k: summary[k]["p50_ms"] or 0))
        if local:
            print(f"\n--- {task} -- local, CPU, no network: NOT comparable to the rows above ---")
            _rows_table(summary, local)

        print("\nvalidity, by kind, and the arm each model ran in:")
        for key in hosted + local:
            info = (config.get("models") or {}).get(summary[key]["model"], {})
            arm = info.get("structured_mode", "unknown")
            temp = info.get("temperature", "n/a")
            print(f"  {key:<22} {str(summary[key]['validity']):<58} {arm}, temperature={temp}")

        gold = any(summary[k]["accuracy"] is not None for k in keys)
        if not gold:
            print("\nno ground truth for this task, so agreement instead of accuracy:")
            names = [summary[k]["model"] for k in hosted + local]
            picks = {n: choices(records, n, task) for n in names}
            print(f"{'':<10}" + "".join(f"{n:>10}" for n in names))
            for a in names:
                cells = "".join(
                    f"{'-':>10}" if a == b else
                    f"{stats.agreement(picks[a], picks[b])[0]:>10.2f}"
                    if stats.agreement(picks[a], picks[b])[0] is not None else f"{'n/a':>10}"
                    for b in names)
                print(f"{a:<10}{cells}")

    overhead = config.get("option_overhead") or {}
    if overhead:
        print("\n=== what the option list alone costs in an LLM prompt ===")
        print(f"{'model/task':<22} {'options':>8} {'prompt':>8} {'no opts':>8} {'option tok':>11} {'share':>7}")
        for key, o in overhead.items():
            share = (o["option_tokens"] / o["prompt_tokens_with_options"]
                     if o.get("option_tokens") and o.get("prompt_tokens_with_options") else None)
            print(f"{key:<22} {o['n_options']:>8} {_fmt(o.get('prompt_tokens_with_options')):>8} "
                  f"{_fmt(o.get('prompt_tokens_without_options')):>8} {_fmt(o.get('option_tokens')):>11} "
                  f"{_fmt(share, '{:.0%}'):>7}")

    total = sum(s["cost_usd"] or 0 for s in summary.values())
    print(f"\ntotal recorded spend: ${total:.4f} over {len(records)} calls")


if __name__ == "__main__":
    main()

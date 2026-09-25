"""Reads the call log and prints the tables.

    python -m typed_decisions.report --tag pilot

Two things are kept apart everywhere, and both are in the grouping key so they
cannot be merged by accident:

  * **Transport.** This experiment ran over the Vercel AI Gateway before it ran
    over OpenRouter. Different provider, different route, different models, so the
    numbers are not comparable and a row is per (model, task, transport).
  * **Laya**, which ran on this machine's CPU with no network in it. Its latency
    is not comparable to a hosted call, so it is printed in its own group and
    labelled.

The `--repeat` pass is excluded from the measured numbers and reported on its own
as a disagreement rate, because folding a repeated subset into the run's latency
and cost would weight those examples twice.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from . import stats, store

RUN_DIR = Path("typed_decisions/runs")
LOCAL = {"laya"}  # not comparable to a hosted call; grouped separately everywhere
MEASURED_PASS = 0


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def group_key(record):
    """Model, task and **transport**. The transport is in the key so OpenRouter and
    the older Vercel records can never end up averaged into one row."""
    return f"{record['model']}/{record['task']}@{record.get('transport', 'unknown')}"


def _measured(records):
    return [r for r in records if r.get("repeat", MEASURED_PASS) == MEASURED_PASS]


def summarise(records):
    """-> {"<model>/<task>@<transport>": row}. Nothing is dropped before scoring: an
    answer that could not be parsed is a wrong answer, not a missing one.

    The repeat pass is excluded -- it is a determinism probe over a subset, not
    more of the sample."""
    groups = defaultdict(list)
    for r in _measured(records):
        groups[group_key(r)].append(r)

    out = {}
    for name, rows in groups.items():
        timed = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
        walls = [r["wall_ms"] for r in rows if r["wall_ms"] is not None]
        validity = defaultdict(int)
        for r in rows:
            validity[r["validity"]] += 1
        costs = [r["cost"] for r in rows if r.get("cost") is not None]
        correct = [r["correct"] for r in rows if r["correct"] is not None]
        tokens_in = [r["input_tokens"] for r in rows if r["input_tokens"] is not None]
        p50, p95 = stats.percentile(timed, 50), stats.percentile(timed, 95)
        accuracy = (sum(correct) / len(correct)) if correct else None
        cost_per_call = (sum(costs) / len(rows)) if costs else None
        out[name] = {
            "model": rows[0]["model"],
            "model_id": rows[0]["model_id"],
            "task": rows[0]["task"],
            "transport": rows[0].get("transport", "unknown"),
            "n_options": rows[0]["n_options"],
            "structured": rows[0]["structured"],
            "local": rows[0]["model"] in LOCAL,
            "n": len(rows),
            "n_timed": len(timed),
            "p50_ms": p50,
            "p95_ms": p95,
            "tail_ratio": stats.tail_ratio(p95, p50),
            "wall_p50_ms": stats.percentile(walls, 50),
            "wall_p95_ms": stats.percentile(walls, 95),
            "mean_input_tokens": _mean(tokens_in),
            "mean_output_tokens": _mean([r["output_tokens"] for r in rows]),
            "n_tokens_reported": len(tokens_in),
            "cost_usd": sum(costs) if costs else None,
            "cost_per_call_usd": cost_per_call,
            "cost_per_1k_usd": (cost_per_call * 1000) if cost_per_call is not None else None,
            "cost_per_correct_usd": stats.cost_per_correct_decision(cost_per_call, accuracy),
            "validity": dict(validity),
            "valid": validity["valid"],
            "valid_rate": validity["valid"] / len(rows),
            "accuracy": accuracy,
            "accuracy_ci": stats.bootstrap_ci(correct) if correct else None,
            "retries": sum(r["retries"] for r in rows),
            "failed": sum(1 for r in rows if r["failed"]),
        }
    return out


def determinism(records):
    """-> {"<model>/<task>@<transport>": {disagreement_rate, n}} over the repeated
    subset only.

    A decision model should be exactly reproducible; an LLM at temperature 0 often
    is not. Only models that actually ran a second pass appear, because a zero here
    would otherwise read as perfect determinism when nothing was measured."""
    first, second = defaultdict(dict), defaultdict(dict)
    for r in records:
        bucket = second if r.get("repeat", MEASURED_PASS) != MEASURED_PASS else first
        bucket[group_key(r)][r["example_id"]] = r["choice"]

    out = {}
    for name, repeated in second.items():
        rate, n = stats.disagreement_rate(first.get(name, {}), repeated)
        if n:
            out[name] = {"disagreement_rate": rate, "n": n}
    return out


def identical_prompt_tokens(config, task):
    """What each model is billed in input tokens for **one identical prompt**.

    Taken from the option-overhead probe, which sends every model the same example
    with the same words, so the spread is each model's tokenizer and protocol
    overhead rather than a difference in the prompt. Models that report no tokens
    at all are left out rather than shown as zero."""
    overhead = (config or {}).get("option_overhead") or {}
    rows = [(key, o.get("prompt_tokens_with_options"))
            for key, o in overhead.items()
            if key.endswith(f"/{task}") and o.get("prompt_tokens_with_options") is not None]
    return sorted(rows, key=lambda kv: kv[1])


def choices(records, model, task):
    return {r["example_id"]: r["choice"] for r in _measured(records)
            if r["model"] == model and r["task"] == task}


def _fmt(value, spec="{:.0f}"):
    return "n/a" if value is None else spec.format(value)


def _rows_table(summary, keys):
    print(f"{'model':<9} {'model id':<30} {'n':>4} {'p50 ms':>7} {'p95 ms':>7} {'tail':>5} "
          f"{'wall p50':>8} {'in tok':>7} {'$/1k':>8} {'$/correct':>10} {'valid':>6} {'acc':>16} {'rtry':>5}")
    for key in keys:
        s = summary[key]
        acc = "n/a (no gold)" if s["accuracy"] is None else \
            f"{s['accuracy']:.3f} [{s['accuracy_ci'][0]:.2f},{s['accuracy_ci'][1]:.2f}]"
        print(f"{s['model']:<9} {s['model_id'][:30]:<30} {s['n']:>4} {_fmt(s['p50_ms']):>7} "
              f"{_fmt(s['p95_ms']):>7} {_fmt(s['tail_ratio'], '{:.1f}'):>5} "
              f"{_fmt(s['wall_p50_ms']):>8} {_fmt(s['mean_input_tokens']):>7} "
              f"{_fmt(s['cost_per_1k_usd'], '{:.4f}'):>8} "
              f"{_fmt(s['cost_per_correct_usd'], '{:.6f}'):>10} "
              f"{s['valid_rate']:>5.0%} {acc:>16} {s['retries']:>5}")


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

    transports = sorted({s["transport"] for s in summary.values()})
    if len(transports) > 1:
        print(f"!! this log holds more than one transport ({', '.join(transports)}). "
              f"They are reported separately and must not be compared.")

    for task in dict.fromkeys(r["task"] for r in records):
        keys = [k for k, s in summary.items() if s["task"] == task]
        hosted = [k for k in keys if not summary[k]["local"]]
        local = [k for k in keys if summary[k]["local"]]
        n_options = summary[keys[0]]["n_options"]
        for transport in sorted({summary[k]["transport"] for k in hosted}):
            rows = [k for k in hosted if summary[k]["transport"] == transport]
            print(f"\n=== {task} ({n_options} options) -- hosted, via {transport} ===")
            _rows_table(summary, sorted(rows, key=lambda k: summary[k]["p50_ms"] or 0))
        if local:
            print(f"\n--- {task} -- local, CPU, no network: NOT comparable to the rows above ---")
            _rows_table(summary, local)

        print("\nvalidity, by kind, and the arm each model ran in:")
        for key in hosted + local:
            info = (config.get("models") or {}).get(summary[key]["model"], {})
            arm = info.get("structured_mode", "unknown")
            temp = info.get("temperature", "n/a")
            print(f"  {key:<34} {str(summary[key]['validity']):<52} {arm}, temperature={temp}")

        tokens = identical_prompt_tokens(config, task)
        if tokens:
            print(f"\ninput tokens for one identical prompt ({task}) -- same words for every "
                  f"model, so the spread is tokenizer and protocol overhead:")
            for key, n in tokens:
                print(f"  {key:<26} {n:>6}")
            if len(tokens) > 1:
                print(f"  -> {tokens[-1][0]} is billed {tokens[-1][1] / tokens[0][1]:.1f}x "
                      f"the input tokens of {tokens[0][0]} for the same text")

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

    repeats = determinism(records)
    if repeats:
        print("\n=== determinism: the same example asked twice ===")
        print("measured on a SUBSET of the run, not on all of it; n is that subset.")
        print(f"{'model/task@transport':<34} {'n':>5} {'disagreement':>13}")
        for key, d in sorted(repeats.items(), key=lambda kv: kv[1]["disagreement_rate"]):
            print(f"  {key:<32} {d['n']:>5} {d['disagreement_rate']:>12.0%}")

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

    total = sum(r["cost"] for r in records if r.get("cost") is not None)
    print(f"\ntotal recorded spend: ${total:.4f} over {len(records)} calls "
          f"(including {len(records) - len(_measured(records))} repeat-pass calls)")


if __name__ == "__main__":
    main()

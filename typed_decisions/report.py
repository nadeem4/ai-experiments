"""Reads the call log and prints the tables. Runs nothing.

    python -m typed_decisions.report --tag pilot
    python -m typed_decisions.report --tag pilot --models jev,laya,gpt

**Every number here comes out of the store.** `--models` computes the full table
over any named subset, so a model added next month is compared against today's
numbers without re-running a thing -- and the property that makes that sound is
that a subset report equals the full report restricted to the same models. No
number is computed across models.

That is only valid if every model saw byte-identical inputs, so the report
**refuses to mix records whose run-spec hashes differ**, naming what changed. A
regenerated example list or a reworded instruction fails loudly here rather than
producing a table that looks entirely normal and is quietly meaningless.

Three things are kept apart everywhere, all of them in a grouping key so they
cannot be merged by accident:

  * **Transport.** This experiment ran over the Vercel AI Gateway before it ran
    over OpenRouter. Different provider, route and models, so a row is per
    (model, task, transport).
  * **Laya**, which runs on this machine's CPU with no network in it. Its latency
    is not comparable to a hosted call, so it is printed in its own group and
    labelled.
  * **The arms.** Only `main` feeds latency, cost, accuracy and validity. The
    position-bias arms re-ask examples already counted there and are reported
    entirely separately, so no example is weighted twice. The `--repeat` pass is
    excluded for the same reason and reported as a disagreement rate.

The figures are written from the same results structure the tables are printed
from, by default, so a chart can never drift from the number beside it.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from . import catalog, spec as spec_module, stats, store

RUN_DIR = Path("typed_decisions/runs")
RESULTS_DIR = Path("typed_decisions/results")
MEASURED_PASS = 0
MEASURED_ARM = "main"


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def group_key(record):
    """Model, task and **transport**. The transport is in the key so OpenRouter and
    the older Vercel records can never end up averaged into one row."""
    return f"{record['model']}/{record['task']}@{record.get('transport', 'unknown')}"


def _measured(records):
    """The headline sample: the main arm, first pass.

    The position-bias arms re-ask examples that are already in the main arm with
    the options in another order. Folding them in would weight those examples six
    or seven times over in the latency, cost and accuracy of the run, so they are
    excluded here and reported entirely separately."""
    return [r for r in records
            if r.get("repeat", MEASURED_PASS) == MEASURED_PASS
            and r.get("arm", MEASURED_ARM) == MEASURED_ARM]


def _bias(records):
    return [r for r in records
            if r.get("repeat", MEASURED_PASS) == MEASURED_PASS
            and spec_module.is_bias(r.get("arm", MEASURED_ARM))]


def _select(records, models):
    """Restrict to a named subset, refusing a name with no records.

    An empty row for a model nobody ran would read as a model that answered
    nothing, which is a different and much worse claim than 'not measured'."""
    if models is None:
        return records
    present = {r["model"] for r in records}
    missing = [m for m in models if m not in present]
    if missing:
        raise SystemExit(f"no records for {', '.join(missing)} in this store. "
                         f"It holds: {', '.join(sorted(present)) or '(nothing)'}. "
                         f"Run them first -- `run --models {','.join(missing)}` fills only "
                         f"what is missing.")
    wanted = set(models)
    return [r for r in records if r["model"] in wanted]


def require_one_spec(records, specs):
    """-> {task: spec hash}, or refuse, naming what moved.

    **This is the safety property of the whole design.** Comparing a model
    measured today against one measured next month is only sound if both saw
    byte-identical inputs. A regenerated example list or a reworded instruction
    produces a table that looks entirely normal and is quietly invalid, so the
    report will not print one: it stops, names the hashes, says which models are
    on which, and diffs the two specs field by field.

    The granularity is **per task**, because one spec per task is the design --
    the option list, the instruction and the example ids all differ between
    `ag_news` and `clinc150`. The rule is that every record *of one task* was
    produced under that task's spec.

    A record with no spec hash at all is refused for the same reason. It cannot
    be shown to have seen the same inputs, and unknown is not a free pass."""
    if not records:
        raise SystemExit("no records to report on")
    by_task = defaultdict(lambda: defaultdict(set))
    for r in records:
        by_task[r["task"]][r.get("spec_hash")].add(r["model"])

    out = {}
    for task, by_hash in sorted(by_task.items()):
        if len(by_hash) == 1:
            only = next(iter(by_hash))
            if only is None:
                raise SystemExit(f"every {task} record carries an unknown spec hash: this "
                                 f"store predates the frozen run spec and cannot be "
                                 f"compared against one.")
            if only not in specs:
                raise SystemExit(f"the {task} records were produced under spec "
                                 f"{spec_module.short(only)}, whose spec file is not in this "
                                 f"run directory. Without it the inputs cannot be checked, "
                                 f"so the report stops.")
            out[task] = only
            continue

        lines = [f"the {task} records come from more than one run spec, and they must not "
                 f"be mixed into one table:"]
        for digest, models in sorted(by_hash.items(), key=lambda kv: str(kv[0])):
            label = spec_module.short(digest) if digest else "unknown (pre-spec records)"
            lines.append(f"  {label}: {', '.join(sorted(models))}")
        known = [h for h in by_hash if h in specs]
        if len(known) >= 2:
            first, second = known[0], known[1]
            lines.append(f"what differs between {spec_module.short(first)} and "
                         f"{spec_module.short(second)}:")
            lines += [f"  - {line}" for line in spec_module.diff(specs[first], specs[second])]
        else:
            lines.append("at least one of those specs is not in this run directory, so the "
                         "difference cannot be shown -- only that there is one.")
        lines.append("re-run the models that are behind, or report on one spec at a time.")
        raise SystemExit("\n".join(lines))
    return out


def summarise(records, models=None):
    """-> {"<model>/<task>@<transport>": row}. Nothing is dropped before scoring: an
    answer that could not be parsed is a wrong answer, not a missing one.

    `models` computes the full table over any named subset, purely from the
    store, running nothing. The property that makes the design work is that a
    subset report equals the full report restricted to the same models -- adding
    a model next month cannot move an existing model's number by a hair, because
    no number here is computed across models.

    The repeat pass and the position-bias arms are both excluded -- one is a
    determinism probe over a subset, the other re-asks examples already counted
    here, and folding either in would weight those examples twice or more."""
    groups = defaultdict(list)
    for r in _measured(_select(records, models)):
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
            "local": catalog.is_local(rows[0]["model"]),
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
            "spec_hash": rows[0].get("spec_hash"),
            # Measured, not declared: a model that is supposed to hand back a
            # distribution and did not must show as not, or this column is
            # documentation rather than a measurement.
            "returns_probability": any(r.get("probabilities") for r in rows),
            "declares_probability": catalog.returns_probability(rows[0]["model"])
                                    if rows[0]["model"] in catalog.ALL else None,
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
        if r.get("arm", MEASURED_ARM) != MEASURED_ARM:
            continue  # the bias arms vary the order on purpose; that is not drift
        bucket = second if r.get("repeat", MEASURED_PASS) != MEASURED_PASS else first
        bucket[group_key(r)][r["example_id"]] = r["choice"]

    out = {}
    for name, repeated in second.items():
        rate, n = stats.disagreement_rate(first.get(name, {}), repeated)
        if n:
            out[name] = {"disagreement_rate": rate, "n": n}
    return out


# --- position bias -----------------------------------------------------------


def bias_key(record):
    """Model and task, without the transport.

    Unlike latency and cost, a flip rate is not a property of the wire: it is the
    share of examples whose answer changed when only the option order changed. It
    is still kept per task, because 4 options and 151 are the whole question."""
    return f"{record['model']}/{record['task']}"


def position_bias(records, models=None):
    """-> {"<model>/<task>": {flip, gold, positions, n_examples}}.

    Three numbers, because they answer three different questions.

      * **flip** -- the share of the subset whose answer is not the same across
        all three seeded orderings. Same example, same options, only the order
        changes, so every flip is pure error. It says *how much* a model moves.
      * **gold** -- accuracy with the correct option placed deliberately first,
        middle and last, and the spread between them. This is the causal measure:
        a flip rate alone cannot tell you which positions are favoured.
      * **positions** -- the distribution of where in the shown list the model's
        answer sat. It says what *kind* of bias the model has. A model that
        always answers with whatever is listed first has a high flip rate and a
        completely characteristic distribution; a model that is merely noisy has
        the same flip rate and a flat one.

    Models with no position-bias calls are absent rather than reported as having
    no bias, which is a claim the run did not make."""
    groups = defaultdict(list)
    for r in _bias(_select(records, models)):
        groups[bias_key(r)].append(r)

    out = {}
    for name, rows in groups.items():
        random_answers = defaultdict(list)
        placements = {p: [] for p in spec_module.PLACEMENTS}
        positions, n_options = [], rows[0].get("n_options") or 0
        for r in sorted(rows, key=lambda r: (r["example_id"], r["arm"])):
            kind, _, which = r["arm"].partition(":")
            if kind == spec_module.RANDOM_PREFIX:
                random_answers[r["example_id"]].append(r["choice"])
            elif which in placements:
                placements[which].append(r["correct"])
            shown = r.get("options_shown")
            if shown:
                positions.append(stats.chosen_position(r["choice"], shown))
        n_orders = max((len(v) for v in random_answers.values()), default=0)
        out[name] = {
            "model": rows[0]["model"],
            "task": rows[0]["task"],
            "n_options": n_options,
            "n_examples": len({r["example_id"] for r in rows}),
            "n_calls": len(rows),
            # A model that could not answer at all -- Laya cannot fit 151 options
            # into its head budget -- scores zero at every gold position, and that
            # is a structural failure rather than a position effect. The count is
            # carried so the table and the figure can say so instead of drawing it.
            "n_valid": sum(1 for r in rows if r.get("choice") is not None),
            "flip": stats.flip_rate(random_answers, n_orders=n_orders or 1),
            "gold": stats.accuracy_by_position(placements),
            "positions": stats.position_histogram(
                positions, n_options=n_options or 1,
                n_bins=min(n_options or 1, 10)),
        }
    return out


# --- paired comparison -------------------------------------------------------


def correctness(records, model, task):
    """{example: 0/1} over the main arm, for one model on one task."""
    return {r["example_id"]: r["correct"] for r in _measured(records)
            if r["model"] == model and r["task"] == task and r["correct"] is not None}


def paired_accuracy(records, a, b, task):
    """-> {"n", "mean_diff", "ci"} for model `a` minus model `b` on `task`.

    Over the **intersection of the examples both models actually answered**, and
    `n` is reported everywhere because that intersection can be far smaller than
    either model's own sample -- one model failing half its calls is exactly when
    a paired comparison is most tempting and least representative.

    Paired rather than two intervals: some examples are simply harder, and that
    shared per-item variation swamps the effect otherwise. Two overlapping
    intervals say nothing about how two models compare to each other."""
    return {"a": a, "b": b, "task": task,
            **stats.paired(correctness(records, a, task), correctness(records, b, task))}


# --- calibration, for the models that return a probability at all -------------


def _distributions(records, arm):
    """-> {"model/task": [({option: p}, gold)]} for one arm.

    Only records that actually carry a distribution. A model that was supposed to
    return one and did not is absent, not scored."""
    out = defaultdict(list)
    for r in records:
        if r.get("arm", MEASURED_ARM) != arm or r.get("repeat", MEASURED_PASS) != MEASURED_PASS:
            continue
        probabilities = r.get("probabilities")
        if probabilities and r.get("gold") is not None:
            out[bias_key(r)].append((probabilities, r["gold"]))
    return out


def calibrations(records, models=None, n_bins=stats.N_BINS):
    """-> {"<model>/<task>": {temperature, ece_raw, ece_scaled, reliability, ...}}.

    Present only for the models that returned a distribution: an LLM in
    structured mode returns a label, so there is nothing to calibrate and a zero
    here would read as perfectly calibrated.

    **The temperature is fitted on the `validation` arm and never on test.**
    Those rows come out of the dataset's train split; the arm is what keeps the
    two apart inside one store. A model with no validation rows gets a raw ECE
    and no temperature, which is the honest outcome -- fitting on test to fill
    the column in would be the one thing that must never happen."""
    records = _select(records, models)
    test = _distributions(records, MEASURED_ARM)
    validation = _distributions(records, "validation")
    out = {}
    for name, rows in test.items():
        scored = stats.calibrate(validation.get(name, []), rows, n_bins=n_bins)
        if scored is None:
            continue
        scored["model"], scored["task"] = name.split("/", 1)
        scored["reliability"] = stats.reliability(
            [max(p.values()) for p, _ in rows],
            [int(max(p, key=p.get) == gold) for p, gold in rows], n_bins=n_bins)
        out[name] = scored
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


def load_specs(run_dir):
    """Every spec file in the run directory, keyed by hash and hash-checked.

    The report needs them to say *what differs* when a store holds more than one,
    rather than only that something does."""
    out = {}
    for path in sorted(Path(run_dir).glob("spec-*.json")):
        frozen = spec_module.read(path)
        out[frozen["hash"]] = frozen["payload"]
    return out


def build_results(records, config, specs, spec_hashes, models=None):
    """The one results structure the tables and every figure are both built from.

    One source, so a chart can never disagree with the table printed beside it."""
    summary = summarise(records, models)
    return {
        "tag": config.get("tag"),
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "spec_hashes": spec_hashes,
        "spec": {task: {k: (specs.get(digest) or {}).get(k) for k in
                        ("task", "dataset", "config", "split", "seed", "instructions")}
                 for task, digest in spec_hashes.items()},
        "models": sorted({s["model"] for s in summary.values()}),
        "summary": summary,
        "position_bias": position_bias(records, models),
        "determinism": determinism(_select(records, models)),
        "calibration": calibrations(records, models),
        "option_overhead": config.get("option_overhead") or {},
        "tasks": config.get("tasks") or {},
        "model_resolution": config.get("models") or {},
        "total_recorded_spend_usd": sum(r["cost"] for r in _select(records, models)
                                        if r.get("cost") is not None),
        "n_calls": len(_select(records, models)),
    }


def _print_position_bias(bias):
    if not bias:
        return
    print("\n=== position bias: same example, same options, only the order changes ===")
    print("Measured on a SUBSET of the run; n is that subset. These calls are excluded")
    print("from the latency and cost tables above, so no example is weighted twice.")
    print(f"{'model/task':<24} {'opts':>5} {'n':>4} {'usable':>7} {'flip':>6} "
          f"{'gold 1st':>9} {'gold mid':>9} {'gold last':>10} {'spread':>7} {'mean pos':>9}")
    for key, b in sorted(bias.items(), key=lambda kv: (kv[1]["task"], kv[1]["model"])):
        accuracy = b["gold"]["accuracy"]
        print(f"{key:<24} {b['n_options']:>5} {b['n_examples']:>4} "
              f"{b['n_valid']:>3}/{b['n_calls']:<3} "
              f"{_fmt(b['flip']['rate'], '{:.0%}'):>6} "
              f"{_fmt(accuracy.get('first'), '{:.3f}'):>9} "
              f"{_fmt(accuracy.get('middle'), '{:.3f}'):>9} "
              f"{_fmt(accuracy.get('last'), '{:.3f}'):>10} "
              f"{_fmt(b['gold']['spread'], '{:.3f}'):>7} "
              f"{_fmt(b['positions']['mean_normalised'], '{:.2f}'):>9}")
    print("mean pos: 0 = always picks whatever is listed first, 1 = always the last,")
    print("0.5 = no positional preference. It says what kind of bias, not how much.")
    dead = [k for k, b in bias.items() if b["n_valid"] == 0]
    if dead:
        print(f"!! {', '.join(sorted(dead))} returned no usable answer under ANY ordering. "
              f"Its zeroes are a failure, not a position effect, and it has no flip rate.")


def _print_calibration(scored):
    if not scored:
        print("\nno model in this run returned a probability, so nothing can be calibrated")
        print("and no confidence threshold is available from any of them.")
        return
    print("\n=== calibration, for the models that return a probability at all ===")
    print("The temperature was fitted on a validation split carved out of TRAIN.")
    print(f"{'model/task':<24} {'n test':>7} {'n val':>6} {'T':>6} "
          f"{'ECE raw':>9} {'ECE scaled':>11}")
    for key, c in sorted(scored.items()):
        print(f"{key:<24} {c['n_test']:>7} {c['n_validation']:>6} "
              f"{_fmt(c['temperature'], '{:.2f}'):>6} {_fmt(c['ece_raw'], '{:.4f}'):>9} "
              f"{_fmt(c['ece_scaled'], '{:.4f}'):>11}")
    refused = [k for k, c in scored.items() if c.get("fit_refused")]
    if refused:
        minimum = next(iter(scored.values())).get("min_validation")
        print(f"!! no temperature was fitted for {', '.join(sorted(refused))}: fewer than "
              f"{minimum} validation rows. On a handful of confident, correct rows the fit "
              f"runs to the bottom of its range and returns a scaled ECE of zero, which is "
              f"an artefact of the sample and not a calibration. Raw ECE only.")


def _print_paired(records, summary, task):
    """Every model against the most accurate one, paired, over the intersection."""
    keys = [k for k, s in summary.items() if s["task"] == task and s["accuracy"] is not None]
    if len(keys) < 2:
        return
    best = max(keys, key=lambda k: summary[k]["accuracy"])
    leader = summary[best]["model"]
    print(f"\npaired against {leader}, the most accurate model on {task}:")
    print("per-item differences over the examples BOTH models answered, not two intervals.")
    print(f"{'model':<12} {'n shared':>9} {'mean diff':>10} {'95% interval':>22}")
    for key in sorted(keys, key=lambda k: -summary[k]["accuracy"]):
        model = summary[key]["model"]
        if model == leader:
            continue
        out = paired_accuracy(records, leader, model, task)
        interval = ("n/a" if out["ci"][0] is None
                    else f"[{out['ci'][0]:+.3f}, {out['ci'][1]:+.3f}]")
        crosses = "" if out["ci"][0] is None or out["ci"][0] > 0 else "   (crosses zero)"
        print(f"{model:<12} {out['n']:>9} {_fmt(out['mean_diff'], '{:+.3f}'):>10} "
              f"{interval:>22}{crosses}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="pilot")
    p.add_argument("--models", default=None,
                   help="comma-separated subset to report on. Computed entirely from the "
                        "store: this command never calls a model.")
    args = p.parse_args(argv)

    run_dir = RUN_DIR / args.tag
    records = store.load(run_dir / "calls.jsonl")
    if not records:
        raise SystemExit(f"no calls recorded under {run_dir}")
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    config["tag"] = args.tag
    models = [m.strip() for m in args.models.split(",")] if args.models else None

    specs = load_specs(run_dir)
    spec_hashes = require_one_spec(_select(records, models), specs)
    for task, digest in spec_hashes.items():
        print(f"{task}: run spec {spec_module.short(digest)} -- every record below was "
              f"produced under it.")

    results = build_results(records, config, specs, spec_hashes, models)
    summary = results["summary"]

    transports = sorted({s["transport"] for s in summary.values()})
    if len(transports) > 1:
        print(f"!! this log holds more than one transport ({', '.join(transports)}). "
              f"They are reported separately and must not be compared.")

    for task in dict.fromkeys(s["task"] for s in summary.values()):
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

        print("\nvalidity by kind, the arm each model ran in, and whether it returns "
              "a probability:")
        for key in hosted + local:
            info = (config.get("models") or {}).get(summary[key]["model"], {})
            mode = info.get("structured_mode", "unknown")
            temp = info.get("temperature", "n/a")
            prob = "distribution" if summary[key]["returns_probability"] else "label only"
            print(f"  {key:<34} {str(summary[key]['validity']):<44} {mode}, "
                  f"temperature={temp}, {prob}")

        tokens = identical_prompt_tokens(config, task)
        if tokens:
            print(f"\ninput tokens for one identical prompt ({task}) -- same words for every "
                  f"model, so the spread is tokenizer and protocol overhead:")
            for key, n in tokens:
                print(f"  {key:<26} {n:>6}")
            if len(tokens) > 1:
                print(f"  -> {tokens[-1][0]} is billed {tokens[-1][1] / tokens[0][1]:.1f}x "
                      f"the input tokens of {tokens[0][0]} for the same text")

        _print_paired(_select(records, models), summary, task)

    _print_position_bias(results["position_bias"])
    _print_calibration(results["calibration"])

    repeats = results["determinism"]
    if repeats:
        print("\n=== determinism: the same example asked twice ===")
        print("measured on a SUBSET of the run, not on all of it; n is that subset.")
        print(f"{'model/task@transport':<34} {'n':>5} {'disagreement':>13}")
        for key, d in sorted(repeats.items(), key=lambda kv: kv[1]["disagreement_rate"]):
            print(f"  {key:<32} {d['n']:>5} {d['disagreement_rate']:>12.0%}")

    overhead = results["option_overhead"]
    if overhead:
        print("\n=== what the option list alone costs in an LLM prompt ===")
        print(f"{'model/task':<24} {'options':>8} {'prompt':>8} {'no opts':>8} "
              f"{'option tok':>11} {'share':>7}")
        for key, o in overhead.items():
            share = (o["option_tokens"] / o["prompt_tokens_with_options"]
                     if o.get("option_tokens") and o.get("prompt_tokens_with_options") else None)
            print(f"{key:<24} {o['n_options']:>8} "
                  f"{_fmt(o.get('prompt_tokens_with_options')):>8} "
                  f"{_fmt(o.get('prompt_tokens_without_options')):>8} "
                  f"{_fmt(o.get('option_tokens')):>11} {_fmt(share, '{:.0%}'):>7}")
        negative = [k for k, o in overhead.items()
                    if (o.get("option_tokens") or 0) < 0]
        if negative:
            print(f"!! {', '.join(sorted(negative))} reported FEWER prompt tokens with the "
                  f"option list than without it. That is the provider's accounting, not a "
                  f"measurement of the options, and the figure for it is not usable.")

    selected = _select(records, models)
    print(f"\ntotal recorded spend: ${results['total_recorded_spend_usd']:.4f} over "
          f"{results['n_calls']} calls "
          f"({len(selected) - len(_measured(selected))} of them outside the measured arm)")

    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / f"{args.tag}.json"
    results_path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {results_path}")

    # Always, so a chart cannot drift from the table it was printed beside.
    from . import figures

    out = figures.write_all(results, results_dir / "figures")
    for name in out["written"]:
        print(f"  figure: {results_dir / 'figures' / name}")
    for name, reason in out["skipped"].items():
        print(f"  figure {name}: NOT drawn -- {reason}")


if __name__ == "__main__":
    main()

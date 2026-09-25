"""Many models, one job, two tasks, and a frozen run spec. The CLI.

    python -m typed_decisions.run --models all --tasks ag_news --limit 50 --tag pilot

Every call appends one line to `<run-dir>/calls.jsonl` holding the exact request,
the exact response, the provider's own usage block **and the hash of the run
spec**. A restarted run skips whatever is already there.

## The design this file exists to serve

Models will be added later, and they must be comparable against today's numbers
without re-running anything. That is only sound if every model saw byte-identical
inputs, so `spec.py` freezes them -- the dataset, the split, the ordered example
ids from the seeded shuffle, the instruction text, the option texts and every
position-bias ordering -- and hashes the lot.

  * `run --models <names>` fills only the records missing for those models
    against the current spec, and runs nothing else.
  * `report --models a,b,c` computes the full table over any subset, from the
    store, running nothing.
  * The report **refuses** to mix records whose spec hashes differ.

Adding a hosted model is one line in `catalog.py` (or `--models name=exact/id`
on the command line). Adding a local one is one adapter module and one line.

## The arms

  * `main` -- the canonical option order, every example. The only arm that counts
    towards latency, cost, accuracy and validity.
  * `rand:0..2` -- the bias subset under three seeded orderings, identical across
    every model, for the flip rate.
  * `gold:first|middle|last` -- the bias subset with the correct option placed
    deliberately at each position, for accuracy by position and the spread.
  * `validation` -- rows from **train**, run only for the models that return a
    probability, and used only to fit the one temperature in `stats.py`.
    Nothing is ever fitted on test.

Nothing here hardcodes a model id: OpenRouter is asked what it hosts, the catalog
is ranked current-generation-first, and the ranked candidates are probed with one
tiny call until one answers, because the catalog lists far more models than an
account is entitled to call.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

from . import (catalog, laya_local, openrouter, prompts, spec as spec_module,
               stats, store, tasks)

RUN_DIR = Path("typed_decisions/runs")
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
WARMUP_CALLS = 20
SPEND_THRESHOLD_USD = 1.00  # above this, the run needs --yes-spend
VALIDATION_ARM = "validation"


def api_key():
    """Read through code, never printed and never copied anywhere."""
    if key := os.environ.get("OPENROUTER_API_KEY"):
        return key
    lines = ROOT_ENV.read_text().splitlines() if ROOT_ENV.exists() else []
    return next((l.split("=", 1)[1].strip() for l in lines
                 if l.startswith("OPENROUTER_API_KEY=")), None)


def prober(key):
    """One five-token call. A model in the catalog this account cannot call answers
    403, which is the only way to tell entitlement from listing."""
    def probe(model_id):
        try:
            openrouter._http_post(
                openrouter.CHAT_URL,
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                {"model": model_id, "max_tokens": 16,
                 "messages": [{"role": "user", "content": "hi"}]})
            return True
        except Exception:
            return False
    return probe


def build_models(wanted, key, verbose=True):
    """-> ({name: model}, resolution).

    The resolution records every id that was tried and whether the account could
    reach it, so a family that could only be reached at an older generation says
    so rather than passing as a current comparison."""
    listing = catalog.fetch(key)
    # The decision models are absent from the default listing and appear only
    # under ?output_modalities=decisions, so Jev is looked up in its own catalog.
    decisions = catalog.fetch_decisions(key)
    probe = prober(key)
    models, resolution = {}, {}
    for name, pinned in wanted.items():
        kind = catalog.kind_of(name)

        if kind == "local":
            adapter = catalog.load_adapter(catalog.LOCAL[name]["adapter"])
            models[name] = adapter()
            resolution[name] = {"model_id": models[name].model_id, "route": "local (cpu)",
                                "transport": laya_local.TRANSPORT, "structured_mode": "typed",
                                "returns_probability": catalog.returns_probability(name),
                                "tried": []}
            continue

        if kind == "decision":
            model_id = pinned or catalog.DECISION[name]["model_id"]
            # The catalog lists the floating minor, not the dated build, so the
            # version block is looked up under either and the *pinned* id is what
            # goes on the wire.
            entry = (catalog.find(decisions, model_id)
                     or catalog.find(decisions, model_id.rsplit("-", 1)[0]))
            if entry is None:
                resolution[name] = {"model_id": None,
                                    "unavailable": "no decision model in the OpenRouter catalog"}
                continue
            models[name] = openrouter.JevModel(name, model_id, api_key=key)
            resolution[name] = {"model_id": model_id,
                                "route": catalog.DECISION[name]["route"],
                                "transport": openrouter.TRANSPORT, "structured_mode": "typed",
                                "returns_probability": catalog.returns_probability(name),
                                "generation": catalog.version(entry).get("released"),
                                "version": catalog.version(entry), "tried": []}
            continue

        if pinned:
            entry, tried = catalog.find(listing, pinned), [(pinned, True)]
            if entry is None:
                raise SystemExit(f"{pinned} is not in the OpenRouter catalog")
        else:
            entry, tried = catalog.resolve(listing, catalog.HOSTED[name], probe)
        if verbose:
            for model_id, ok in tried:
                print(f"  {name}: {model_id} {'reachable' if ok else 'REFUSED by this account'}")
        if entry is None:
            resolution[name] = {"model_id": None, "tried": tried,
                                "unavailable": "no model in this family is reachable by "
                                               "this account"}
            continue
        # Sent only where the catalog says the model accepts it, recorded either way.
        temperature = 0 if catalog.accepts_temperature(entry) else None
        reasoning_effort = "minimal" if catalog.accepts_reasoning_effort(entry) else None
        models[name] = openrouter.ChatModel(name, entry["id"], api_key=key,
                                            temperature=temperature,
                                            reasoning_effort=reasoning_effort)
        version = catalog.version(entry)
        refused = [m for m, ok in tried if not ok]
        resolution[name] = {"model_id": entry["id"], "route": "POST /api/v1/chat/completions",
                            "transport": openrouter.TRANSPORT,
                            "structured_mode": "json_schema (strict)", "temperature": temperature,
                            "reasoning_effort": reasoning_effort,
                            "returns_probability": catalog.returns_probability(name),
                            "generation": version.get("released"),
                            "current_generation": not refused,
                            "refused_above_it": refused,
                            "version": version, "tried": tried}
    return models, resolution


def overhead_probe(model, task):
    """What the option list alone costs in the prompt, measured rather than counted.

    The same prompt is sent twice, once with the option block and once without,
    and the provider's own reported prompt tokens are differenced. The second call
    keeps structured mode on and only takes the options out of it -- dropping the
    schema entirely would charge the structured-output scaffolding to the options,
    which some providers bill as a tool definition."""
    example = task["examples"][0]
    with_options = model.decide(example["text"], task["instructions"], task["criteria"])
    if isinstance(model, openrouter.ChatModel):
        body = dict(with_options["request"])
        body["messages"] = [{"role": "user", "content":
                             prompts.render(task["instructions"], example["text"],
                                            task["criteria"], with_options=False)}]
        if "response_format" in body:
            body["response_format"] = prompts.schema(task["criteria"], with_enum=False)
        bare = model._call(openrouter.CHAT_URL, body)
        bare_tokens = ((bare["response"] or {}).get("usage") or {}).get("prompt_tokens")
    else:
        bare_tokens = None
    full = with_options["input_tokens"]
    return {
        "model_id": model.model_id,
        "n_options": len(task["criteria"]),
        "prompt_tokens_with_options": full,
        "prompt_tokens_without_options": bare_tokens,
        "option_tokens": ((full - bare_tokens)
                          if (full is not None and bare_tokens is not None) else None),
        "output_tokens": with_options["output_tokens"],
        "cost_one_call": with_options["cost"],
    }


def plan_run(jobs, warmup, probes):
    """The spend estimate, from measured tokens and the catalog's price. Only the
    estimate uses a price list; every cost that gets *reported* is OpenRouter's
    own per-call `cost`."""
    plan = []
    for (name, task_name), pending in jobs.items():
        probe = probes.get((name, task_name)) or {}
        per_call = probe.get("cost_one_call")
        plan.append({"model": f"{name}/{task_name}", "calls": pending,
                     "warmup": warmup if pending else 0,
                     "cost_per_call": per_call or 0.0,
                     "measured": per_call is not None})
    return plan


def arm_rows(task, arm):
    """The example rows this arm covers, in spec order."""
    if arm == VALIDATION_ARM:
        return list(task["validation"])
    wanted = set(spec_module.examples_for(task["spec"], arm))
    return [e for e in task["examples"] if e["id"] in wanted]


def criteria_for(task, arm, example_id):
    """The option map to send, in this arm's order.

    The validation arm uses the canonical order: it exists to fit a temperature,
    not to measure position."""
    if arm in (spec_module.MEASURED_ARM, VALIDATION_ARM):
        return task["criteria"]
    return spec_module.options_in_order(task["spec"], arm, example_id)


def arms_to_run(task, selection, models):
    """-> [(arm, [model names])].

    The validation arm runs only for models that return a probability: it exists
    to fit a temperature, and an LLM that hands back a label has nothing to fit.
    Paying for those calls would buy nothing."""
    out = []
    if "main" in selection:
        out.append((spec_module.MEASURED_ARM, list(models)))
    if "bias" in selection:
        for arm in spec_module.arms(task["spec"]):
            if spec_module.is_bias(arm):
                out.append((arm, list(models)))
    if "validation" in selection and task["validation"]:
        probabilistic = [m for m in models if catalog.returns_probability(m)]
        if probabilistic:
            out.append((VALIDATION_ARM, probabilistic))
    return out


def run_arm(model, name, task, arm, path, warmup=0, repeat=0):
    """One pass over one arm. Every record carries the spec hash, the arm and the
    exact option order that went over the wire."""
    spec_hash = task["spec"]["hash"]
    rows = arm_rows(task, arm)
    done = store.done_keys(path)
    todo = store.pending(rows, spec_hash, name, task["name"], arm, done, repeat=repeat)
    label = f"{name} x {task['name']} [{arm}]" + (f" pass {repeat}" if repeat else "")
    print(f"\n{label}: {len(rows)} examples, "
          f"{len(rows) - len(todo)} already recorded, {len(todo)} to run")
    if not todo:
        return

    # Warm-ups come from outside the measured window where there are examples to
    # spare, so provider-side prompt caching cannot make a measured call look
    # faster or cheaper than a cold one. Only the main arm is warmed: the bias
    # arms measure which option a model picks, not how fast it picks it.
    if arm == spec_module.MEASURED_ARM and not repeat and warmup:
        warm = task["warmup_pool"][:warmup] or rows[:warmup]
        print(f"  warming up ({len(warm)} calls, not recorded"
              f"{', reusing measured examples' if not task['warmup_pool'] else ''})")
        for row in warm:
            model.decide(row["text"], task["instructions"], task["criteria"])

    started = time.perf_counter()
    for i, row in enumerate(todo, start=1):
        criteria = criteria_for(task, arm, row["id"])
        out = model.decide(row["text"], task["instructions"], criteria)
        if out["structured"] == "typed":
            verdict = ({"choice": out["choice"], "validity": "valid", "detail": None}
                       if out["choice"] in criteria
                       else {"choice": None,
                             "validity": "api_error" if out["failed"] else "not_an_option",
                             "detail": out["error"] or str(out["choice"])})
        else:
            verdict = prompts.classify(out["content"], criteria)
            if out["failed"]:
                verdict = {"choice": None, "validity": "api_error", "detail": out["error"]}
        store.append(path, {
            "spec_hash": spec_hash,
            "arm": arm,
            "model": name,
            "model_id": model.model_id,
            "task": task["name"],
            "example_id": row["id"],
            "n_options": len(criteria),
            # The order that actually went over the wire, so the chosen-position
            # distribution is read off what was shown rather than off the
            # canonical order this arm deliberately permuted.
            "options_shown": list(criteria),
            "gold": row["label"],
            "choice": verdict["choice"],
            "validity": verdict["validity"],
            "validity_detail": verdict["detail"],
            "correct": int(verdict["choice"] == row["label"]),
            "latency_ms": out["latency_ms"],
            "wall_ms": out["wall_ms"],
            "input_tokens": out["input_tokens"],
            "output_tokens": out["output_tokens"],
            "reasoning_tokens": out.get("reasoning_tokens"),
            "cost": out["cost"],
            "retries": out["retries"],
            "failed": out["failed"],
            "error": out["error"],
            "structured": out["structured"],
            "temperature": out["temperature"],
            "reasoning_effort": out.get("reasoning_effort"),
            # The transport is on every record, so a local CPU number and a
            # network round trip can never be averaged together.
            "transport": out["transport"],
            "repeat": repeat,
            "probabilities": out.get("probabilities"),
            "request": out["request"],
            "response": out["response"],
        })
        if i % 10 == 0 or i == len(todo):
            rate = (time.perf_counter() - started) / i
            print(f"  {i}/{len(todo)}  {rate:.2f}s/call  "
                  f"eta {(len(todo) - i) * rate / 60:.1f} min")


def write_config(path, resolution, probes, args, models, loaded):
    """Merged into whatever is already there, so a run assembled over several
    invocations -- one model at a time, or resumed after a failure -- ends up with
    every model and every overhead measurement in one file."""
    import numpy

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = {
        "experiment": "many models, one job: what a typed decision costs at 4 options and 151",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "transport": openrouter.TRANSPORT,
        "transport_note": "every hosted call in this run went over OpenRouter. Earlier runs of "
                          "this experiment went over the Vercel AI Gateway and their numbers "
                          "are not comparable: different provider, route and models. The "
                          "transport is on every call record and in the report's grouping key.",
        "routes": {"llm": openrouter.CHAT_URL, "decision": openrouter.SYSTEMONE_URL},
        "seed": args.seed,
        "warmup_calls": args.warmup,
        "repeat": args.repeat,
        "spec_hashes": {name: task["spec"]["hash"] for name, task in loaded.items()},
        "tasks": {name: {"dataset": task["spec"]["payload"]["dataset"],
                         "config": task["spec"]["payload"]["config"],
                         "split": task["spec"]["payload"]["split"],
                         "licence": task["licence"],
                         "source": task["source"],
                         "n_examples": len(task["examples"]),
                         "n_options": len(task["criteria"]),
                         "n_validation": len(task["validation"]),
                         "bias_subset": task["spec"]["payload"]["position_bias"]["subset_size"],
                         "n_random_orders":
                             task["spec"]["payload"]["position_bias"]["n_random_orders"]}
                  for name, task in loaded.items()},
        "last_invocation": {"tasks": args.tasks, "limit": args.limit, "models": args.models,
                            "arms": args.arms, "probe_only": args.probe_only,
                            "repeat": args.repeat},
        "models": {**(previous.get("models") or {}), **resolution},
        "option_overhead": {**(previous.get("option_overhead") or {}),
                            **{f"{m}/{t}": v for (m, t), v in probes.items()}},
        "answer_directive": prompts.ANSWER_DIRECTIVE,
        "question_id": openrouter.QUESTION_ID,
        "max_tokens": openrouter.MAX_TOKENS,
        "calibration": {
            "n_bins": stats.N_BINS,
            "min_validation": stats.MIN_VALIDATION,
            "fitted_on": "a validation split carved out of TRAIN, stratified by label. "
                         "Nothing is ever fitted on test.",
        },
        "fairness": {
            "structured_mode": "json_schema with a strict enum of the option keys; "
                               "recorded per model",
            "temperature": "0 where the model accepts one, null where it does not, "
                           "recorded per model",
            "reasoning_effort": "'minimal' where the model advertises the parameter, null "
                                "where it does not",
            "retries": "transport only (429/5xx); an unparseable answer is never re-asked",
            "latency": "the successful attempt only; wall_ms carries the backoff",
            "model_choice": "current generation first, cheapest tier within it; every id "
                            "refused above the one that answered is recorded per model",
            "identical_inputs": "every model reads its examples, its instruction text and "
                                "its option orderings off one frozen run spec, whose hash "
                                "is on every call record",
        },
        "versions": {"numpy": numpy.__version__, "python": platform.python_version()},
        "hardware": {
            "platform": platform.platform(),
            "note": "laya runs on this machine's CPU -- no usable GPU -- so its latency is "
                    "not comparable to a hosted call and is grouped separately everywhere",
        },
    }
    if "laya" in models:
        try:
            payload["versions"]["laya"] = __import__("laya").__version__
            payload["models"]["laya"]["cfg"] = models["laya"].config()
        except Exception as e:  # laya is optional: the hosted arms must still run
            payload["versions"]["laya"] = f"unavailable: {e}"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="all",
                   help=f"comma-separated: {', '.join(catalog.ALL)}, or name=model-id to pin")
    p.add_argument("--tasks", default="ag_news", help=f"comma-separated: {', '.join(tasks.NAMES)}")
    p.add_argument("--limit", type=int, default=0, help="first N examples per task (0 = all)")
    p.add_argument("--tag", default="pilot", help="run directory under typed_decisions/runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=WARMUP_CALLS)
    p.add_argument("--arms", default="main,bias,validation",
                   help="which arms to run: main, bias, validation")
    p.add_argument("--bias-subset", type=int, default=tasks.BIAS_SUBSET,
                   help="how many leading examples the position-bias arms cover")
    p.add_argument("--validation", type=int, default=tasks.VALIDATION_N,
                   help="rows carved out of TRAIN to fit the calibration temperature on")
    p.add_argument("--threshold", type=float, default=SPEND_THRESHOLD_USD)
    p.add_argument("--yes-spend", action="store_true",
                   help="confirm a run estimated above --threshold")
    p.add_argument("--probe-only", action="store_true",
                   help="measure the option-token overhead and price the run, then stop")
    p.add_argument("--repeat", type=int, default=0,
                   help="ask the first N examples of each task a second time, to measure "
                        "determinism")
    args = p.parse_args(argv)

    key = api_key()
    if not key:
        raise SystemExit("no OPENROUTER_API_KEY in the environment or the repo-root .env")
    run_dir = RUN_DIR / args.tag
    path = run_dir / "calls.jsonl"
    selection = {a.strip() for a in args.arms.split(",") if a.strip()}

    # The account is capped monthly; a run that would run into the cap should not
    # start. Never printed as anything but a total.
    try:
        account = openrouter.balance(key)
        remaining = account.get("limit_remaining")
        print(f"account: ${remaining:.4f} left of a ${account.get('limit')} monthly cap"
              if remaining is not None else "account: no cap reported")
    except Exception as e:
        remaining = None
        print(f"account: could not read the balance ({type(e).__name__})")

    print("resolving each family against what this account can actually call:")
    models, resolution = build_models(catalog.select(args.models), key)
    for name, info in resolution.items():
        if info.get("unavailable"):
            print(f"  {name}: UNAVAILABLE -- {info['unavailable']}")

    loaded = {name: tasks.load(name, limit=args.limit, seed=args.seed,
                               bias_subset=args.bias_subset, validation_n=args.validation)
              for name in args.tasks.split(",")}
    print("\nthe frozen run spec for each task:")
    for name, task in loaded.items():
        frozen = task["spec"]
        spec_module.write(run_dir / f"spec-{name}-{spec_module.short(frozen['hash'])}.json",
                          frozen)
        print(f"  {name}: spec {spec_module.short(frozen['hash'])}, "
              f"{len(task['examples'])} examples, {len(task['criteria'])} options, "
              f"bias subset {frozen['payload']['position_bias']['subset_size']}, "
              f"{len(task['validation'])} validation rows from train")

    print("\nmeasuring the option-token overhead (two calls per model per task):")
    config_path = run_dir / "config.json"
    previous = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    measured = previous.get("option_overhead") or {}
    probes, jobs = {}, {}
    done = store.done_keys(path)
    for task_name, task in loaded.items():
        planned = arms_to_run(task, selection, models)
        for name, model in models.items():
            pending = sum(
                len(store.pending(arm_rows(task, arm), task["spec"]["hash"], name, task_name,
                                  arm, done))
                for arm, who in planned if name in who)
            jobs[(name, task_name)] = pending
            if pending or f"{name}/{task_name}" not in measured:
                probes[(name, task_name)] = overhead_probe(model, task)
                over = probes[(name, task_name)]["option_tokens"]
                print(f"  {name}/{task_name}: {len(task['criteria'])} options, "
                      f"{over if over is not None else 'n/a'} of "
                      f"{probes[(name, task_name)]['prompt_tokens_with_options']} prompt tokens")

    plan = plan_run(jobs, args.warmup, probes)
    estimate = stats.estimate(plan)
    print("\nestimated spend for this run (measured tokens x the catalog's price):")
    for row in sorted(plan, key=lambda r: -estimate["by_model"][r["model"]]):
        if row["calls"]:
            print(f"  {row['model']:<24} {row['calls']:>6} calls + {row['warmup']} warm-up  "
                  f"${estimate['by_model'][row['model']]:.4f}"
                  f"{'' if row['measured'] else '  (no price reported)'}")
    print(f"  {'TOTAL':<24} ${estimate['total_usd']:.4f}")
    if remaining is not None:
        print(f"  account balance before this run: ${remaining:.4f}")
    if args.probe_only:
        write_config(config_path, resolution, probes, args, models, loaded)
        print(f"\nprobe only: nothing measured, wrote {config_path}")
        return
    if stats.needs_confirmation(estimate["total_usd"], args.threshold) and not args.yes_spend:
        raise SystemExit(f"\nestimated ${estimate['total_usd']:.2f} is over the "
                         f"${args.threshold:.2f} threshold: re-run with --yes-spend to confirm")
    if remaining is not None and estimate["total_usd"] > remaining:
        raise SystemExit(f"\nestimated ${estimate['total_usd']:.2f} is more than the "
                         f"${remaining:.2f} left on the account's monthly cap")

    for task_name, task in loaded.items():
        for arm, who in arms_to_run(task, selection, models):
            for name in who:
                run_arm(models[name], name, task, arm, path, warmup=args.warmup)

    # The repeat pass runs only after every first pass is done, so no model is
    # asked the same example twice back to back off a warm cache.
    if args.repeat:
        print(f"\n--- repeat pass: the first {args.repeat} examples of each task, again, "
              f"to measure determinism ---")
        for task_name, task in loaded.items():
            trimmed = dict(task, examples=task["examples"][:args.repeat])
            for name, model in models.items():
                run_arm(model, name, trimmed, spec_module.MEASURED_ARM, path, repeat=1)

    write_config(config_path, resolution, probes, args, models, loaded)
    print(f"\nwrote {path} and {config_path}")


if __name__ == "__main__":
    main()

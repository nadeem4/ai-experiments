"""What one typed decision costs, across model families. The task and the CLI.

    python -m decision_cost.run --tasks highway --limit 10 --tag pilot

Every (model, task, example) appends one line to `<run-dir>/calls.jsonl` holding
the exact request, the exact response and the provider's own usage block, and a
restarted run skips whatever is already there.

Nothing here hardcodes a model id: OpenRouter is asked what it hosts, the catalog
is ranked current-generation-first per family, and the ranked candidates are probed
with one tiny call until one answers, because the catalog lists far more models
than an account is entitled to call. Whatever answered goes into `config.json`,
together with every id that was refused -- so a family that could only be reached
at an older generation says so rather than passing as a current comparison.

`--repeat N` asks the first N examples of each task a second time. Both answers
stay in the log, keyed by pass, and the report turns them into a disagreement rate
per model: a decision model should be exactly reproducible, an LLM at temperature
zero often is not.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

from . import catalog, laya_local, openrouter, parse, prompts, stats, store, tasks

RUN_DIR = Path("decision_cost/runs")
ROOT_ENV = Path(__file__).resolve().parents[1] / ".env"
WARMUP_CALLS = 20
SPEND_THRESHOLD_USD = 1.00  # above this, the run needs --yes-spend
ARMS = ("gpt", "claude", "gemini", "gemma", "jev", "laya")


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


def parse_arms(spec):
    """`gpt,claude` picks arms; `gpt=openai/gpt-5.4` pins one to an exact id, which
    is how you point the experiment at a better model once the account can reach it."""
    if spec == "all":
        return {name: None for name in ARMS}
    out = {}
    for item in spec.split(","):
        name, _, pinned = item.strip().partition("=")
        if name not in ARMS:
            raise SystemExit(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
        out[name] = pinned or None
    return out


def build_models(arms, key, verbose=True):
    """-> ({name: model}, resolution) -- the resolution records every id that was
    tried and whether the account could reach it."""
    listing = catalog.fetch(key)
    # The decision models are absent from the default listing and appear only
    # under ?output_modalities=decisions, so Jev is looked up in its own catalog.
    decisions = catalog.fetch_decisions(key)
    probe = prober(key)
    models, resolution = {}, {}
    for name, pinned in arms.items():
        if name == "laya":
            models[name] = laya_local.LayaModel()
            resolution[name] = {"model_id": models[name].model_id, "route": "local (cpu)",
                                "transport": laya_local.TRANSPORT,
                                "structured_mode": "typed", "tried": []}
            continue
        if name == "jev":
            jev_id = pinned or catalog.JEV_ID
            entry = catalog.find(decisions, jev_id) or catalog.find(decisions, "typesafe/jev-1.13")
            if entry is None:
                resolution[name] = {"model_id": None,
                                    "unavailable": "no decision model in the OpenRouter catalog"}
                continue
            models[name] = openrouter.JevModel(name, jev_id, api_key=key)
            resolution[name] = {"model_id": jev_id, "route": "POST /api/v1/systemone",
                                "transport": openrouter.TRANSPORT,
                                "structured_mode": "typed", "generation": catalog.version(entry).get("released"),
                                "version": catalog.version(entry), "tried": []}
            continue
        if pinned:
            entry, tried = catalog.find(listing, pinned), [(pinned, True)]
            if entry is None:
                raise SystemExit(f"{pinned} is not in the OpenRouter catalog")
        else:
            entry, tried = catalog.resolve(listing, catalog.FAMILIES[name], probe)
        if verbose:
            for model_id, ok in tried:
                print(f"  {name}: {model_id} {'reachable' if ok else 'REFUSED by this account'}")
        if entry is None:
            resolution[name] = {"model_id": None, "tried": tried,
                                "unavailable": "no model in this family is reachable by this account"}
            continue
        # Sent only where the catalog says the model accepts it, recorded either
        # way. The current GPT and Claude tiers take neither.
        temperature = 0 if catalog.accepts_temperature(entry) else None
        reasoning_effort = "minimal" if catalog.accepts_reasoning_effort(entry) else None
        models[name] = openrouter.ChatModel(name, entry["id"], api_key=key,
                                            temperature=temperature,
                                            reasoning_effort=reasoning_effort)
        version = catalog.version(entry)
        # The whole ladder, so a family reached only at an older generation is
        # visible as such rather than passing as a current comparison.
        refused = [m for m, ok in tried if not ok]
        resolution[name] = {"model_id": entry["id"], "route": "POST /api/v1/chat/completions",
                            "transport": openrouter.TRANSPORT,
                            "structured_mode": "json_schema (strict)", "temperature": temperature,
                            "reasoning_effort": reasoning_effort,
                            "generation": version.get("released"),
                            "current_generation": not refused,
                            "refused_above_it": refused,
                            "version": version, "tried": tried}
    return models, resolution


def overhead_probe(model, task):
    """What the option list alone costs in the prompt, measured rather than counted.

    The same prompt is sent twice, once with the option block and once without, and
    the provider's own reported prompt tokens are differenced. It is the structural
    point of the experiment: options are free for a decision model and are paid for
    on every LLM call.

    The second call keeps structured mode on and only takes the options out of it --
    both the option block in the prompt and the enum in the schema. Dropping the
    schema entirely would charge the structured-output scaffolding to the options,
    which some providers bill as a tool definition."""
    example = task["examples"][0]
    with_options = model.decide(example["state"], task["instructions"], task["criteria"])
    if isinstance(model, openrouter.ChatModel):
        body = dict(with_options["request"])
        body["messages"] = [{"role": "user", "content":
                             prompts.render(task["instructions"], example["state"], task["criteria"],
                                            with_options=False)}]
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
        "option_tokens": (full - bare_tokens) if (full is not None and bare_tokens is not None) else None,
        "output_tokens": with_options["output_tokens"],
        "cost_one_call": with_options["cost"],
    }


def plan_run(jobs, warmup, probes):
    """The spend estimate, from measured tokens and the catalog's price. Only the
    estimate uses a price list; every cost that gets *reported* is OpenRouter's own
    per-call `cost`."""
    plan = []
    for (name, task_name), pending in jobs.items():
        probe = probes.get((name, task_name)) or {}
        per_call = probe.get("cost_one_call")
        plan.append({"model": f"{name}/{task_name}", "calls": pending, "warmup": warmup if pending else 0,
                     "cost_per_call": per_call or 0.0,
                     "measured": per_call is not None})
    return plan


def run_pair(model, name, task, path, warmup, repeat=0):
    """One pass over a task. `repeat=0` is the measured run; `repeat=1` is the
    determinism probe over the first `--repeat` examples, recorded under its own
    key so both answers survive and neither overwrites the other."""
    examples = task["examples"] if not repeat else task["examples"][:task["repeat_n"]]
    done = store.done_keys(path)
    todo = store.pending(examples, name, task["name"], done, repeat=repeat)
    label = f"{name} x {task['name']}" + (f" (repeat pass {repeat})" if repeat else "")
    print(f"\n{label}: {len(examples)} examples, "
          f"{len(examples) - len(todo)} already recorded, {len(todo)} to run")
    if not todo:
        return
    # Warm-ups come from outside the measured window where there are examples to
    # spare, so that provider-side prompt caching cannot make a measured call look
    # faster or cheaper than a cold one. The repeat pass is deliberately not warmed
    # again: it is measuring reproducibility, not speed.
    if not repeat:
        warm = task["warmup_pool"][:warmup] or examples[:warmup]
        print(f"  warming up ({len(warm)} calls, not recorded"
              f"{', reusing measured examples' if not task['warmup_pool'] else ''})")
        for row in warm:
            model.decide(row["state"], task["instructions"], task["criteria"])

    started = time.perf_counter()
    for i, row in enumerate(todo, start=1):
        out = model.decide(row["state"], task["instructions"], task["criteria"])
        if out["structured"] == "typed":
            verdict = ({"choice": out["choice"], "validity": "valid", "detail": None}
                       if out["choice"] in task["criteria"]
                       else {"choice": None, "validity": "api_error" if out["failed"] else "not_an_option",
                             "detail": out["error"] or str(out["choice"])})
        else:
            verdict = parse.classify(out["content"], task["criteria"])
            if out["failed"]:
                verdict = {"choice": None, "validity": "api_error", "detail": out["error"]}
        store.append(path, {
            "model": name,
            "model_id": model.model_id,
            "task": task["name"],
            "example_id": row["id"],
            "n_options": len(task["criteria"]),
            "gold": row["gold"],
            "choice": verdict["choice"],
            "validity": verdict["validity"],
            "validity_detail": verdict["detail"],
            "correct": None if row["gold"] is None else int(verdict["choice"] == row["gold"]),
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
            # The transport is on every record, so OpenRouter numbers and the
            # older Vercel-gateway ones can never be averaged together.
            "transport": out["transport"],
            "repeat": repeat,
            "probabilities": out.get("probabilities"),
            "request": out["request"],
            "response": out["response"],
        })
        if i % 10 == 0 or i == len(todo):
            rate = (time.perf_counter() - started) / i
            print(f"  {i}/{len(todo)}  {rate:.2f}s/call  eta {(len(todo) - i) * rate / 60:.1f} min")


def write_config(path, resolution, probes, args, models):
    """Merged into whatever is already there, so a run assembled over several
    invocations -- one arm at a time, or resumed after a failure -- ends up with
    every model and every overhead measurement in one file."""
    import numpy

    path = Path(path)
    # A --probe-only run records no calls, so nothing else has created the run
    # directory yet: this writer cannot assume the store ran first.
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload = {
        "experiment": "what one typed decision costs, across model families",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "transport": openrouter.TRANSPORT,
        "transport_note": "every hosted call in this run went over OpenRouter. Earlier runs of "
                          "this experiment went over the Vercel AI Gateway and their numbers are "
                          "not comparable: different provider, route and models. The transport is "
                          "on every call record and in the report's grouping key.",
        "routes": {"llm": openrouter.CHAT_URL, "decision": openrouter.SYSTEMONE_URL},
        "seed": args.seed,
        "warmup_calls": args.warmup,
        "repeat": args.repeat,
        # A run can be assembled over several invocations; `models` and
        # `option_overhead` accumulate, these are the last command that touched it.
        "last_invocation": {"tasks": args.tasks, "limit": args.limit, "models": args.models,
                            "probe_only": args.probe_only, "repeat": args.repeat},
        "models": {**(previous.get("models") or {}), **resolution},
        "option_overhead": {**(previous.get("option_overhead") or {}),
                            **{f"{m}/{t}": v for (m, t), v in probes.items()}},
        "answer_directive": prompts.ANSWER_DIRECTIVE,
        "question_id": openrouter.QUESTION_ID,
        "max_tokens": openrouter.MAX_TOKENS,
        "fairness": {
            "structured_mode": "json_schema with a strict enum of the option keys; "
                               "recorded per model",
            "temperature": "0 where the model accepts one, null where it does not, recorded per model",
            "reasoning_effort": "'minimal' where the model advertises the parameter, null where it "
                                "does not; the current tiers are reasoning models and their "
                                "thinking tokens can exhaust a small answer budget",
            "retries": "transport only (429/5xx); an unparseable answer is never re-asked",
            "latency": "the successful attempt only; wall_ms carries the backoff",
            "model_choice": "current generation first, cheapest tier within it; every id refused "
                            "above the one that answered is recorded per model",
        },
        "versions": {
            "numpy": numpy.__version__,
            "python": platform.python_version(),
        },
        "hardware": {
            "platform": platform.platform(),
            "note": "laya runs on this machine's CPU -- no usable GPU -- so its latency is not "
                    "comparable to a hosted call and is grouped separately everywhere",
        },
    }
    if "laya" in models:
        try:
            payload["versions"]["laya"] = __import__("laya").__version__
            payload["models"]["laya"]["cfg"] = models["laya"].config()
        except Exception as e:  # laya is optional: the hosted arms must still run
            payload["versions"]["laya"] = f"unavailable: {e}"
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="all", help="comma-separated arms, optionally name=model-id")
    p.add_argument("--tasks", default="highway", help=f"comma-separated: {', '.join(tasks.NAMES)}")
    p.add_argument("--limit", type=int, default=0, help="first N examples per task (0 = all)")
    p.add_argument("--tag", default="pilot", help="run directory under decision_cost/runs")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=WARMUP_CALLS)
    p.add_argument("--threshold", type=float, default=SPEND_THRESHOLD_USD)
    p.add_argument("--yes-spend", action="store_true", help="confirm a run estimated above --threshold")
    p.add_argument("--probe-only", action="store_true",
                   help="measure the option-token overhead and price the run, then stop")
    p.add_argument("--repeat", type=int, default=0,
                   help="ask the first N examples of each task a second time, to measure "
                        "determinism; both answers are kept and the report gives a "
                        "disagreement rate over that subset")
    args = p.parse_args(argv)

    key = api_key()
    if not key:
        raise SystemExit("no OPENROUTER_API_KEY in the environment or the repo-root .env")
    run_dir = RUN_DIR / args.tag
    path = run_dir / "calls.jsonl"

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
    models, resolution = build_models(parse_arms(args.models), key)
    for name, info in resolution.items():
        if info.get("unavailable"):
            print(f"  {name}: UNAVAILABLE -- {info['unavailable']}")

    loaded = {name: tasks.load(name, args.seed) for name in args.tasks.split(",")}
    for task in loaded.values():
        cut = args.limit or len(task["examples"])
        task["warmup_pool"] = task["examples"][cut:]
        task["examples"] = task["examples"][:cut]
        task["repeat_n"] = min(args.repeat, len(task["examples"]))

    print("\nmeasuring the option-token overhead (two calls per model per task):")
    config_path = run_dir / "config.json"
    previous = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    measured = previous.get("option_overhead") or {}
    probes, jobs = {}, {}
    done = store.done_keys(path)
    for task_name, task in loaded.items():
        for name, model in models.items():
            pending = len(store.pending(task["examples"], name, task_name, done))
            jobs[(name, task_name)] = pending
            # Measured when there is work to price, and also when a finished pair has
            # never been measured -- so the headline number is never missing from a
            # run that was assembled over several invocations.
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
            print(f"  {row['model']:<22} {row['calls']:>5} calls + {row['warmup']} warm-up  "
                  f"${estimate['by_model'][row['model']]:.4f}"
                  f"{'' if row['measured'] else '  (no price reported)'}")
    print(f"  {'TOTAL':<22} ${estimate['total_usd']:.4f}")
    if args.probe_only:
        write_config(config_path, resolution, probes, args, models)
        print(f"\nprobe only: nothing measured, wrote {config_path}")
        return
    if stats.needs_confirmation(estimate["total_usd"], args.threshold) and not args.yes_spend:
        raise SystemExit(f"\nestimated ${estimate['total_usd']:.2f} is over the ${args.threshold:.2f} "
                         f"threshold: re-run with --yes-spend to confirm")

    if remaining is not None and estimate["total_usd"] > remaining:
        raise SystemExit(f"\nestimated ${estimate['total_usd']:.2f} is more than the "
                         f"${remaining:.2f} left on the account's monthly cap")

    for task_name, task in loaded.items():
        for name, model in models.items():
            run_pair(model, name, task, path, args.warmup)

    # The repeat pass runs only after every first pass is done, so no model is
    # asked the same example twice back to back off a warm cache.
    if args.repeat:
        print(f"\n--- repeat pass: the first {args.repeat} examples of each task, again, "
              f"to measure determinism ---")
        for task_name, task in loaded.items():
            for name, model in models.items():
                run_pair(model, name, task, path, args.warmup, repeat=1)

    write_config(config_path, resolution, probes, args, models)
    print(f"\nwrote {path} and {config_path}")


if __name__ == "__main__":
    main()

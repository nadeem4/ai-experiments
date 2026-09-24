"""The task and the CLI. Inference only -- nothing here trains anything.

    python -m banking77.run --arms A-default --limit 200

Every (arm, example) appends one line to `<run-dir>/decisions.jsonl` holding the
exact request, the exact response and the full probability distribution, and a
restarted run skips whatever is already there.
"""
import argparse
import json
import os
import platform
import time
from pathlib import Path

from banking77 import arms as arms_mod
from banking77 import budget, data, hierarchy, options, store

RUN_DIR = Path("banking77/runs")
CACHE_DIR = Path("banking77/.cache")
OPTIONS_FILE = Path("banking77/option_texts/options.json")
WEIGHTS = os.environ.get("LAYA_PATH", r"C:\projects\jev_demo\arena\models\laya")

INSTRUCTIONS = "Which banking intent does this customer message express?"
COARSE_INSTRUCTIONS = "Which area of banking is this customer message about?"
VAL_FRAC = 0.1
WARMUP_CALLS = 20


def question(instructions, texts):
    return {"intent": {"type": "choice", "instructions": instructions, "criteria": options.criteria(texts)}}


def diagnose(agent, instructions, texts, head_max_len, max_len):
    """The truncation diagnostics, read off the sequence laya itself builds."""
    from laya.common import build_sequence

    q = {"t": "choice", "ins": instructions, "crit": options.criteria(texts)}
    ids, markers = build_sequence(agent.tok, "a customer message", q, max_len, head_max_len)
    spans = budget.option_spans(ids, markers, agent.tok.sep_token_id)
    decoded = [agent.tok.decode(span[1:]) for span in spans]
    full_lengths = [
        1 + len(agent.tok(" " + t, add_special_tokens=False)["input_ids"][:48]) for t in texts
    ]
    stats = budget.truncation_stats(spans, full_lengths, decoded)
    stats["head_max_len"] = head_max_len
    stats["max_len"] = max_len
    stats["sequence_tokens"] = len(ids)
    stats["projected_tokens_per_option"] = budget.laya_tokens_per_option(head_max_len, len(texts))
    stats["options_as_the_model_sees_them"] = decoded
    collisions = {}
    for text, seen in zip(texts, decoded):
        collisions.setdefault(seen, []).append(text)
    stats["collisions"] = {k: v for k, v in collisions.items() if len(v) > 1}
    return stats


def load_agent(checkpoint, agents):
    if checkpoint not in agents:
        import laya

        started = time.perf_counter()
        agents[checkpoint] = laya.load(WEIGHTS, subfolder=checkpoint) if checkpoint else laya.load(WEIGHTS)
        print(f"  loaded checkpoint {checkpoint or 'english (root)'} in {time.perf_counter() - started:.1f}s")
    return agents[checkpoint]


def predict_flat(agent, text, texts, by_text):
    q = question(INSTRUCTIONS, texts)
    started = time.perf_counter()
    response = agent.predict(text, q)
    latency_ms = (time.perf_counter() - started) * 1000
    answer = response["answers"]["intent"]
    probs = {by_text[t]: p for t, p in answer["probabilities"].items()}
    return {
        "prediction": by_text[answer["choice"]],
        "probabilities": probs,
        "confidence": answer["confidence"],
        "latency_ms": latency_ms,
        "request": {"state": text, "questions": q},
        "response": response,
    }


def predict_coarse_fine(agent, text, groups):
    coarse_texts = hierarchy.coarse_labels(groups)
    q1 = question(COARSE_INSTRUCTIONS, coarse_texts)
    started = time.perf_counter()
    r1 = agent.predict(text, q1)
    chosen = r1["answers"]["intent"]["choice"]

    fine = hierarchy.fine_labels(groups, chosen)
    fine_by_text = {options.option_text(label): label for label in fine}
    q2 = question(INSTRUCTIONS, list(fine_by_text))
    r2 = agent.predict(text, q2)
    latency_ms = (time.perf_counter() - started) * 1000

    fine_probs = {fine_by_text[t]: p for t, p in r2["answers"]["intent"]["probabilities"].items()}
    coarse_probs = r1["answers"]["intent"]["probabilities"]
    return {
        "prediction": hierarchy.prediction(chosen, fine_probs),
        "probabilities": hierarchy.joint_probabilities(groups, coarse_probs, chosen, fine_probs),
        "confidence": r2["answers"]["intent"]["confidence"],
        "chosen_group": chosen,
        "latency_ms": latency_ms,
        "request": {"state": text, "questions": [q1, q2]},
        "response": [r1, r2],
    }


def arm_examples(spec, test_rows, seed):
    """Arm K restricts the same examples to the labels in its subset; every other
    arm sees the whole test split."""
    labels = options.subset_labels(spec["n_options"], seed)
    keep = set(labels)
    rows = [r for r in test_rows if r["label"] in keep]
    return labels, rows


def run_arm(name, spec, test_rows, val_rows, path, agents, limit, seed):
    labels, test = arm_examples(spec, test_rows, seed)
    _, val = arm_examples(spec, val_rows, seed)
    if limit:
        test, val = test[:limit], val[:limit]
    rows, warmup_rows = test + val, val or test
    done = store.done_keys(path)
    todo = store.pending(rows, name, done)
    texts = options.option_texts(labels)
    by_text = dict(zip(texts, labels))

    print(f"\n{name}: {len(test)} test + {len(val)} validation examples, "
          f"{len(rows) - len(todo)} already recorded, {len(todo)} to run")
    if not todo:
        return None

    agent = load_agent(spec["checkpoint"], agents)
    agent.cfg["head_max_len"] = spec["head_max_len"]
    agent.cfg["max_len"] = spec["max_len"]

    if spec["mode"] == "coarse-fine":
        diagnostics = {
            "coarse": diagnose(agent, COARSE_INSTRUCTIONS, hierarchy.coarse_labels(options.COARSE_GROUPS),
                               spec["head_max_len"], spec["max_len"]),
            "fine": {
                group: diagnose(agent, INSTRUCTIONS, options.option_texts(members),
                                spec["head_max_len"], spec["max_len"])
                for group, members in options.COARSE_GROUPS.items()
            },
        }
    else:
        diagnostics = diagnose(agent, INSTRUCTIONS, texts, spec["head_max_len"], spec["max_len"])
    (path.parent / "diagnostics").mkdir(parents=True, exist_ok=True)
    (path.parent / "diagnostics" / f"{name}.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"  warming up ({WARMUP_CALLS} calls, not recorded)")
    for row in warmup_rows[:WARMUP_CALLS]:
        if spec["mode"] == "coarse-fine":
            predict_coarse_fine(agent, row["text"], options.COARSE_GROUPS)
        else:
            predict_flat(agent, row["text"], texts, by_text)

    started = time.perf_counter()
    for i, row in enumerate(todo, start=1):
        if spec["mode"] == "coarse-fine":
            out = predict_coarse_fine(agent, row["text"], options.COARSE_GROUPS)
        else:
            out = predict_flat(agent, row["text"], texts, by_text)
        store.append(path, {
            "arm": name,
            "example_id": row["id"],
            "split": row["split"],
            "text": row["text"],
            "gold": row["label"],
            "n_options": len(labels),
            "head_max_len": spec["head_max_len"],
            "max_len": spec["max_len"],
            "checkpoint": spec["checkpoint"],
            **out,
        })
        if i % 25 == 0 or i == len(todo):
            rate = (time.perf_counter() - started) / i
            print(f"  {i}/{len(todo)}  {rate:.2f}s/example  eta {(len(todo) - i) * rate / 60:.1f} min")
    return (time.perf_counter() - started) / len(todo)


def write_config(path, names, limit, seed, rates):
    import laya
    import torch
    import transformers

    payload = {
        "experiment": "banking77 option budget",
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "arms": {n: arms_mod.ARMS[n] for n in names},
        "limit": limit,
        "seed": seed,
        "val_frac": VAL_FRAC,
        "warmup_calls": WARMUP_CALLS,
        "instructions": {"flat": INSTRUCTIONS, "coarse": COARSE_INSTRUCTIONS},
        "option_source": "BANKING77 intent labels, underscores replaced by spaces, no descriptions",
        "options_file": str(OPTIONS_FILE),
        "dataset": {"repo": "PolyAI/banking77", "files": f"{data.BASE}/{{train,test}}.csv"},
        "weights": {
            "path": WEIGHTS,
            "checkpoints": {
                "english": "convaiinnovations/laya (repo root)",
                "multilingual": "convaiinnovations/laya subfolder multilingual",
            },
        },
        "versions": {
            "laya": laya.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "hardware": {
            "cpu": platform.processor(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "cuda_available": torch.cuda.is_available(),
            "device_used": "cpu",
            "note": "no usable GPU on this machine; every latency here is CPU and is not "
                    "comparable to Convai's published 33 ms, which is a T4 figure",
        },
        "seconds_per_example": rates,
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arms", default="all", help="comma-separated arm names, or 'all'")
    p.add_argument("--limit", type=int, default=0, help="first N test examples per arm (0 = all)")
    p.add_argument("--tag", default="pilot", help="run directory under banking77/runs")
    p.add_argument("--seed", type=int, default=arms_mod.SEED)
    args = p.parse_args(argv)

    names = arms_mod.resolve(args.arms)
    run_dir = RUN_DIR / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "decisions.jsonl"

    options.freeze(OPTIONS_FILE, options.ALL_LABELS)
    # one permutation, fixed seed, shared by every arm, so `--limit` is a sample of
    # all 77 intents rather than the handful the split happens to start with
    test_rows = data.shuffled(data.load(CACHE_DIR, "test"), args.seed)
    train_rows = data.load(CACHE_DIR, "train")
    _, held_out = data.stratified_split(train_rows, VAL_FRAC, args.seed)
    # they keep their `train-N` ids, because that is what they are
    val_rows = data.shuffled([dict(r, split="val") for r in held_out], args.seed)
    if sorted({r["label"] for r in test_rows}) != options.ALL_LABELS:
        raise SystemExit("the test split's labels do not match the frozen option set")
    print(f"test {len(test_rows)} rows, validation {len(val_rows)} rows held out of train, "
          f"{len(options.ALL_LABELS)} labels")

    agents, rates = {}, {}
    for name in names:
        rate = run_arm(name, arms_mod.ARMS[name], test_rows, val_rows, path, agents, args.limit, args.seed)
        if rate:
            rates[name] = round(rate, 3)
    write_config(run_dir / "config.json", names, args.limit, args.seed, rates)
    print(f"\nwrote {path} and {run_dir / 'config.json'}")


if __name__ == "__main__":
    main()

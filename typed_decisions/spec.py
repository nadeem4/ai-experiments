"""The frozen run spec: everything every model must have seen, and its hash.

The design requirement this file exists for: **models added later must be
comparable against today's numbers without re-running anything.** That is only
sound if every model saw byte-identical inputs, and "byte-identical" has to be
checkable rather than promised. So the spec pins, in one JSON-serialisable
payload:

  * the dataset, the config and the split;
  * the seed, and the ordered list of example ids the seeded shuffle produced;
  * the gold label of each of those examples;
  * the instruction text, verbatim;
  * the option texts, verbatim and in their canonical order;
  * the option ordering every position-bias arm uses, per example, as indices
    into the canonical order;
  * the validation example ids that anything fitted is fitted on.

`hash` is the SHA-256 of that payload serialised canonically. **Every stored call
records it.** `run --models X` fills only the records missing for X against the
current spec. `report --models a,b,c` computes over any subset, from the store,
running nothing -- and refuses to mix records whose spec hashes differ, naming
what differs, because a regenerated example list or a reworded instruction
produces a comparison that is quietly invalid and looks completely normal.

**Why orderings live in the spec rather than being regenerated per model.** A
position-bias arm whose orders came from "the same seed" in each model's process
is one library upgrade away from two models seeing different lists. Here the
orders are data, computed once, hashed, and read back by every arm.

**The arms.**

  * `main` -- the canonical option order, every example. The *only* arm that
    counts towards latency, cost, accuracy and validity, so the position-bias
    calls can never double-weight the examples they cover.
  * `rand:0` / `rand:1` / `rand:2` -- the bias subset under three seeded
    orderings. Same example, same options, only the order changes, so a change
    of answer is pure error. Reported as a flip rate.
  * `gold:first` / `gold:middle` / `gold:last` -- the bias subset with the
    correct option placed deliberately at each end and in the middle. This is
    the causal arm: a flip rate says how much a model moves, only this says
    which positions it favours.
"""
import hashlib
import json
import random

MEASURED_ARM = "main"
RANDOM_PREFIX = "rand"
GOLD_PREFIX = "gold"
PLACEMENTS = ("first", "middle", "last")


def _rng(*parts):
    """A permutation is a function of the spec's own fields and nothing else, so
    a spec rebuilt on another machine, another day, is the same spec."""
    return random.Random("|".join(str(p) for p in parts))


def random_orders(n_options, example_id, seed, n_orders):
    """`n_orders` orderings of the option indices, deterministic in (seed, id).

    Not guaranteed distinct for a tiny option list -- with 4 options and 24
    permutations a collision is ordinary and pretending otherwise would bias the
    orderings. The flip rate is defined over whatever orders were actually used,
    and the spec records them."""
    out = []
    for k in range(n_orders):
        order = list(range(n_options))
        _rng(seed, example_id, "rand", k).shuffle(order)
        out.append(order)
    return out


def gold_orders(n_options, gold_index, example_id, seed):
    """The same options with gold forced to index 0, `n // 2` and `n - 1`.

    The other options are shuffled once, with a seed of their own, and that one
    background order is reused for all three placements -- so the only thing
    that differs between the three calls is where gold sits. If the background
    moved too, a difference in accuracy would not be attributable to position."""
    rest = [i for i in range(n_options) if i != gold_index]
    _rng(seed, example_id, "gold").shuffle(rest)
    middle = n_options // 2
    return {
        "first": [gold_index] + rest,
        "middle": rest[:middle] + [gold_index] + rest[middle:],
        "last": rest + [gold_index],
    }


def build(task, dataset, config, split, seed, examples, instructions, option_texts,
          bias_subset, n_random_orders=3, validation=()):
    """-> {"payload": {...}, "hash": "<sha256>"}.

    `examples` is already shuffled and limited by the caller; the spec records
    the order it was handed, because that order is part of the experiment.
    `bias_subset` is how many of the leading examples the position-bias arms
    cover -- a subset, deliberately, and the size is reported everywhere."""
    canonical = list(option_texts)
    subset = [e["id"] for e in examples[:bias_subset]]
    gold = {e["id"]: e["label"] for e in examples}
    index_of = {name: i for i, name in enumerate(canonical)}
    payload = {
        "task": task,
        "dataset": dataset,
        "config": config,
        "split": split,
        "seed": seed,
        "instructions": instructions,
        "canonical_order": canonical,
        "option_texts": [[name, option_texts[name]] for name in canonical],
        "example_ids": [e["id"] for e in examples],
        "gold": gold,
        "validation_example_ids": [e["id"] for e in validation],
        "validation_gold": {e["id"]: e["label"] for e in validation},
        "position_bias": {
            "n_random_orders": n_random_orders,
            "subset_size": len(subset),
            "subset": subset,
            "random": {eid: random_orders(len(canonical), eid, seed, n_random_orders)
                       for eid in subset},
            "gold": {eid: gold_orders(len(canonical), index_of[gold[eid]], eid, seed)
                     for eid in subset},
        },
    }
    return {"payload": payload, "hash": hash_of(payload)}


def canonical_json(payload):
    """Sorted keys, no incidental whitespace. Two processes that agree on the
    content must agree on the bytes, or the hash is measuring the serialiser."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_of(payload):
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def short(spec_hash):
    """Twelve characters, for tables. Never for a comparison."""
    return (spec_hash or "unknown")[:12]


def arms(spec):
    n = spec["payload"]["position_bias"]["n_random_orders"]
    return ([MEASURED_ARM]
            + [f"{RANDOM_PREFIX}:{k}" for k in range(n)]
            + [f"{GOLD_PREFIX}:{p}" for p in PLACEMENTS])


def is_measured(arm):
    """Only `main` feeds latency, cost, accuracy and validity. The bias arms ask
    about examples that are already in `main`, so folding them in would weight
    those examples several times over."""
    return arm == MEASURED_ARM


def is_bias(arm):
    return arm.startswith((RANDOM_PREFIX + ":", GOLD_PREFIX + ":"))


def examples_for(spec, arm):
    if is_measured(arm):
        return list(spec["payload"]["example_ids"])
    return list(spec["payload"]["position_bias"]["subset"])


def order_for(spec, arm, example_id):
    """The option indices this arm shows for this example, off the spec."""
    payload = spec["payload"]
    if is_measured(arm):
        return list(range(len(payload["canonical_order"])))
    kind, _, which = arm.partition(":")
    if kind == RANDOM_PREFIX:
        return payload["position_bias"]["random"][example_id][int(which)]
    if kind == GOLD_PREFIX:
        return payload["position_bias"]["gold"][example_id][which]
    raise KeyError(f"unknown arm {arm!r}")


def options_in_order(spec, arm, example_id):
    """-> an ordered {option: description} to hand a model, in this arm's order.

    Same words for every model and every arm; only the order moves. That is what
    makes a changed answer attributable to position and nothing else."""
    payload = spec["payload"]
    texts = dict(payload["option_texts"])
    canonical = payload["canonical_order"]
    return {canonical[i]: texts[canonical[i]] for i in order_for(spec, arm, example_id)}


_SUMMARY_FIELDS = ("task", "dataset", "config", "split", "seed", "instructions")


def diff(a, b):
    """-> a list of short human-readable differences, for the report's refusal.

    Short on purpose: the refusal has to name what moved, not print two option
    lists of 151 entries into a terminal."""
    out = []
    for field in _SUMMARY_FIELDS:
        if a.get(field) != b.get(field):
            out.append(f"{field}: {str(a.get(field))[:80]!r} vs {str(b.get(field))[:80]!r}")
    if a.get("canonical_order") != b.get("canonical_order"):
        out.append(f"canonical_order: {len(a.get('canonical_order') or [])} options "
                   f"vs {len(b.get('canonical_order') or [])}, or a different order")
    if a.get("option_texts") != b.get("option_texts"):
        moved = [name for (name, text), (other, other_text)
                 in zip(a.get("option_texts") or [], b.get("option_texts") or [])
                 if (name, text) != (other, other_text)]
        out.append(f"option_texts: {len(moved)} option text(s) differ, first: {moved[:3]}")
    if a.get("example_ids") != b.get("example_ids"):
        first, second = a.get("example_ids") or [], b.get("example_ids") or []
        if len(first) != len(second):
            out.append(f"example_ids: {len(first)} examples vs {len(second)}")
        else:
            at = next(i for i, (x, y) in enumerate(zip(first, second)) if x != y)
            out.append(f"example_ids: same count ({len(first)}) but a different order, "
                       f"first at index {at}")
    if a.get("validation_example_ids") != b.get("validation_example_ids"):
        out.append("validation_example_ids: the validation split differs")
    if a.get("position_bias", {}).get("subset") != b.get("position_bias", {}).get("subset"):
        out.append("position_bias.subset: a different bias subset")
    if a.get("position_bias", {}).get("random") != b.get("position_bias", {}).get("random") \
            or a.get("position_bias", {}).get("gold") != b.get("position_bias", {}).get("gold"):
        out.append("position_bias: the option orderings differ")
    return out


def write(path, spec):
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def read(path):
    """Reads a spec back and **re-checks its hash**, because a stored hash that
    is never verified is a label rather than a proof."""
    from pathlib import Path

    stored = json.loads(Path(path).read_text(encoding="utf-8"))
    recomputed = hash_of(stored["payload"])
    if recomputed != stored["hash"]:
        raise SystemExit(f"{path}: the stored hash {short(stored['hash'])} does not match the "
                         f"payload, which hashes to {short(recomputed)}. The file was edited.")
    return stored

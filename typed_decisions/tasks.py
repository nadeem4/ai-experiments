"""Two tasks, five options and seventy-seven, both built from data that already
exists in this repo or next to it.

The point of the pair is that options are free for a decision model and are charged
for on every LLM call. So the same models answer the same kind of question twice,
once with five options and once with seventy-seven, and the report shows how the
cost of one decision scales with the length of the option list.

  * **highway** -- the arena's recorded highway-env episodes. The states and the
    question, word for word including all five option descriptions, are lifted out
    of `arena/runs/highway/jev/seed-*.json`. No reference policy ships with that
    game -- only `idle` and `random` baselines -- so there is no ground truth and
    the report gives model-to-model agreement instead of accuracy.
  * **banking77** -- the existing harness in `banking77/`. Its loader, its frozen
    option texts and its seeded shuffle are imported rather than reimplemented, so
    the examples are identical to the ones that experiment ran on, and the
    benchmark label is ground truth.
"""
import json
import random
from pathlib import Path

from banking77 import data as b77_data
from banking77 import options as b77_options
from banking77.run import INSTRUCTIONS as BANKING77_INSTRUCTIONS

NAMES = ("highway", "banking77")
ARENA_HIGHWAY = Path(r"C:\projects\jev_demo\arena\runs\highway\jev")
BANKING77_CACHE = Path("banking77/.cache")
BANKING77_OPTIONS = Path("banking77/option_texts/options.json")


def shuffled(rows, seed):
    """One fixed permutation, shared by every model, so `--limit` is a sample.

    Episodes are recorded in step order and BANKING77 ships ordered by label, so an
    unshuffled limit would be ten near-identical highway states or five intents out
    of seventy-seven."""
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def highway_question(events):
    """The instruction line and all five options, taken out of the recording.

    Every episode carries the question it was asked. They must all be the same
    words, or the models in this experiment would not all be answering the same
    question, so a disagreement is refused rather than resolved."""
    questions = [e["questions"]["action"] for e in events if e["type"] == "start"]
    if not questions:
        raise ValueError("no start event: the recording does not carry the question")
    first = questions[0]
    for q in questions[1:]:
        if (q["instructions"], q["criteria"]) != (first["instructions"], first["criteria"]):
            raise ValueError("recorded episodes disagree about the question")
    return first["instructions"], first["criteria"]


def highway_examples(events, seed):
    """Every step of one episode is one example. `gold` is None everywhere: the
    action the recording holds is another model's answer, not a label."""
    return [
        {"id": f"seed-{seed}:t{e['t']}", "state": e["state"], "gold": None}
        for e in events if e["type"] == "step"
    ]


def load_highway(runs_dir=ARENA_HIGHWAY, seed=0):
    paths = sorted(Path(runs_dir).glob("seed-*.json"), key=lambda p: int(p.stem.split("-")[1]))
    if not paths:
        raise SystemExit(f"no recorded highway episodes under {runs_dir}")
    all_events, examples = [], []
    for path in paths:
        events = json.loads(path.read_text(encoding="utf-8"))
        all_events += events
        examples += highway_examples(events, int(path.stem.split("-")[1]))
    instructions, criteria = highway_question(all_events)
    return {
        "name": "highway",
        "instructions": instructions,
        "criteria": criteria,
        "examples": shuffled(examples, seed),
        "has_gold": False,
        "source": str(runs_dir),
    }


def banking77_examples(rows):
    """Gold is the option *text*, because that is the alphabet an answer comes back
    in: the 77 labels with underscores replaced by spaces, and nothing else."""
    return [
        {"id": r["id"], "state": r["text"], "gold": b77_options.option_text(r["label"])}
        for r in rows
    ]


def load_banking77(cache_dir=BANKING77_CACHE, seed=0):
    labels = b77_options.load(BANKING77_OPTIONS) if Path(BANKING77_OPTIONS).exists() else b77_options.ALL_LABELS
    rows = b77_data.shuffled(b77_data.load(cache_dir, "test"), seed)
    return {
        "name": "banking77",
        "instructions": BANKING77_INSTRUCTIONS,
        "criteria": b77_options.criteria(b77_options.option_texts(labels)),
        "examples": banking77_examples(rows),
        "has_gold": True,
        "source": "banking77/ (same loader, same frozen options, same seeded shuffle)",
    }


def load(name, seed=0):
    if name == "highway":
        return load_highway(seed=seed)
    if name == "banking77":
        return load_banking77(seed=seed)
    raise SystemExit(f"unknown task {name!r}; known: {', '.join(NAMES)}")

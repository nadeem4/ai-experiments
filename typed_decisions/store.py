"""One JSONL file of calls, appended and flushed as the run goes.

One line per call holding the exact request sent, the exact response received and
the provider's own usage block. That makes the file both the audit log and the
resume point: whatever is already in it is not called again.

Same shape as `rerank/store.py` and `banking77/store.py`; only the key differs,
which is why it is a separate thirty lines rather than a shared abstraction.
"""
import json
from pathlib import Path


def key(record):
    """(spec hash, model, task, arm, example, pass).

    **The spec hash comes first, and it is the point.** This experiment promises
    that a model added later can be compared against today's numbers without
    re-running anything, and that promise holds only if a changed spec makes
    every existing record stop counting as done. Reword the instruction,
    re-derive the example list, change an option text: the hash moves, nothing
    matches, and the run refills rather than finishing instantly with a table
    that silently mixes two experiments.

    The **arm** is next, because `main` and `gold:last` ask the same example with
    the options in a different order -- two calls, not one.

    The **pass** is what `--repeat` varies: the same example asked a second time
    is a second record, and both answers stay in the log so the disagreement can
    be audited.

    A record written before any of these fields existed reads as an unknown spec
    on the main arm, first pass. An unknown spec is exactly what the report
    refuses to mix with a known one, which is the correct outcome."""
    return (record.get("spec_hash"), record["model"], record["task"],
            record.get("arm", "main"), record["example_id"], record.get("repeat", 0))


def append(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def load(path):
    """Records in the order they were written. A truncated final line -- the
    process was killed mid-write -- is dropped, not raised."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def done_keys(path):
    return {key(r) for r in load(path)}


def pending(rows, spec_hash, model, task, arm, done, repeat=0):
    """The rows this (spec, model, task, arm, pass) has no record for yet.

    `run --models <name>` is exactly this applied to one model: it fills the
    holes for that model against the current spec, and runs nothing else."""
    return [row for row in rows
            if (spec_hash, model, task, arm, row["id"], repeat) not in done]

"""One JSONL file of calls, appended and flushed as the run goes.

One line per (model, task, example) holding the exact request sent, the exact
response received and the provider's own usage block. That makes the file both the
audit log and the resume point: whatever is already in it is not called again.

Same shape as `rerank/store.py` and `banking77/store.py`; only the key differs,
which is why it is a separate twenty lines rather than a shared abstraction.
"""
import json
from pathlib import Path


def key(record):
    """(model, task, example, pass). The pass is what `--repeat` varies: the same
    example asked a second time is a second record, not a duplicate to be skipped,
    and both answers stay in the log so the disagreement can be audited. Records
    written before `--repeat` existed carry no field and read as pass 0."""
    return (record["model"], record["task"], record["example_id"], record.get("repeat", 0))


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


def pending(rows, model, task, done, repeat=0):
    return [row for row in rows if (model, task, row["id"], repeat) not in done]

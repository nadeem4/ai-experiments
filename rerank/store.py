"""One JSONL file of scoring records, appended and flushed as the run goes.

Every (query, passage) the model scored is one line holding the exact request
sent and the exact response received. That makes the file both the audit log and
the resume point: on restart, whatever is already in it is not scored again.
"""
import json
from pathlib import Path


def key(record):
    return (record["method"], record["query_id"], record["doc_id"])


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


def scored_keys(path):
    return {key(r) for r in load(path)}

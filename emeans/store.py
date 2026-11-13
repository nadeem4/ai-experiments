"""One JSONL file of records, appended and flushed as the run goes.

Each record is one unit of work, so the file is both the evidence and the
resume point: on restart, whatever is already in it is not run again.
"""
import json
from pathlib import Path

NL = bytes([10])
KEY_FIELDS = ("part", "dataset", "arm", "setting", "seed")


def key(record):
    """(part, dataset, arm, setting, seed). A missing field is a KeyError."""
    return tuple(record[f] for f in KEY_FIELDS)


def _repair(path):
    """Make the file end on a line boundary before appending.

    A process killed mid-write leaves an unterminated fragment, and appending
    straight after it would glue the next record to it and lose both."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(NL):
        return
    start = data.rfind(NL) + 1
    try:
        json.loads(data[start:])
    except json.JSONDecodeError:
        path.write_bytes(data[:start])  # drop the fragment
    else:
        path.write_bytes(data + NL)  # complete record, only the newline is missing


def append(path, record):
    key(record)  # refuse a record that could never be resumed
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def load(path):
    """Records in the order written. A truncated line is dropped, not raised."""
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

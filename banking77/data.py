"""BANKING77, from PolyAI's own CSVs -- the identical files the Hugging Face
`PolyAI/banking77` loading script downloads.

That repo ships only a Python loading script and has no parquet conversion, so
`datasets` would have to execute remote code to read it. The two CSVs behind the
script are three lines of stdlib away and are the same bytes, so they are fetched
directly and cached next to the run.

The test split (3,080 rows) is what every reported number comes from. The
validation split -- the only thing anything is fitted on -- is carved out of
train, stratified by label, with a fixed seed.
"""
import csv
import io
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
HEADER = ["text", "category"]


def parse_csv(text):
    rows = csv.reader(io.StringIO(text), quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
    header = next(rows, None)
    if header != HEADER:
        raise ValueError(f"unexpected header {header!r}, expected {HEADER!r}")
    return [{"text": row[0], "label": row[1]} for row in rows if row]


def labels_of(rows):
    return sorted({row["label"] for row in rows})


def with_ids(rows, split):
    """`test-0`, `test-1`, ... -- the resume key, so it must not depend on order
    of iteration, filtering or anything an arm chooses."""
    return [dict(row, id=f"{split}-{i}") for i, row in enumerate(rows)]


def stratified_split(rows, val_frac, seed):
    """-> (train, val). Every label contributes `round(n * val_frac)` rows, but
    never fewer than one, so no label is missing from the temperature fit."""
    import random

    by_label = {}
    for i, row in enumerate(rows):
        by_label.setdefault(row["label"], []).append(i)
    held_out = set()
    for label in sorted(by_label):
        group = by_label[label]
        rng = random.Random(f"{seed}:{label}")
        held_out.update(rng.sample(group, max(1, round(len(group) * val_frac))))
    train = [r for i, r in enumerate(rows) if i not in held_out]
    val = [r for i, r in enumerate(rows) if i in held_out]
    return train, val


def shuffled(rows, seed):
    """One fixed permutation, shared by every arm.

    BANKING77 ships its splits ordered by label, so an unshuffled `--limit` would
    hand a pilot five intents out of 77 and call the result accuracy. Shuffling
    once, with a fixed seed and before any arm sees the data, makes a limit a
    sample; sharing the permutation keeps every arm on the same examples."""
    import random

    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def _fetch(split, cache_dir):
    path = Path(cache_dir) / f"{split}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{BASE}/{split}.csv", timeout=120) as r:
            path.write_bytes(r.read())
    return path.read_text(encoding="utf-8")


def load(cache_dir, split):
    """The split as a list of {"id", "split", "text", "label"}, downloaded once."""
    return [dict(r, split=split) for r in with_ids(parse_csv(_fetch(split, cache_dir)), split)]

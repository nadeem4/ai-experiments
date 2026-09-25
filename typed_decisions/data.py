"""AG News and CLINC150, straight from the parquet files the Hugging Face hub
serves, owned by this experiment rather than imported from a sibling one.

Two datasets, chosen to bracket the option count as widely as public labelled
data allows: **4 options against 151**. That is the axis the experiment measures
along, and it is the reason both tasks exist.

  * **AG News** (`fancyzhx/ag_news`) -- 7,600 test rows, four topic labels. The
    hub lists its licence as *unknown*; that is recorded as unknown here and in
    the protocol rather than assumed to be permissive.
  * **CLINC150** (`clinc/clinc_oos`) -- 5,500 test rows, 151 labels: 150 intents
    plus `oos`, the out-of-scope class. CC-BY-3.0. The repo ships three configs
    (`small`, `imbalanced`, `plus`) which differ only in their *train* split and
    share one test split; this experiment uses **`plus`**, because the validation
    carve-out for temperature scaling comes out of train and `plus` is the
    largest of the three.

Three rules this file exists to enforce.

**The label names come out of the file.** Every HF parquet export carries a
`huggingface` blob in its schema metadata holding the `ClassLabel` names in index
order. They are read from there and never written down here, so a dataset that
re-orders or renames its classes cannot silently shift every label by one. A file
without that metadata is an error, not a fallback to integers.

**The id names the row in the file**, not the row's position after some arm
reordered it. `test-0`, `test-1`, ... assigned once at load, before any shuffle,
because that id is the resume key and the run-spec key.

**The shuffle happens before any limit.** CLINC's test split ships sorted by
intent -- its first thirty rows are all `translate` -- so `--limit 50` on the raw
order measures two intents out of 151 and produces a number that looks entirely
reasonable. This repository has made exactly that mistake once. One fixed seed,
one permutation, shared by every model and every arm.
"""
import json
import random

DATASETS = {
    "ag_news": {
        "repo": "fancyzhx/ag_news",
        "config": "default",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
        "text_column": "text",
        "label_column": "label",
        # Verified against the hub on 2026-09-24: the dataset card's metadata
        # says `license: unknown`. Recorded as unknown; this is not a grant.
        "licence": "unknown -- the hub lists `license: unknown` for this dataset. "
                   "Recorded as unknown rather than assumed permissive.",
        "n_test": 7600,
        "n_labels": 4,
    },
    "clinc150": {
        "repo": "clinc/clinc_oos",
        "config": "plus",
        "files": {
            "train": "plus/train-00000-of-00001.parquet",
            "validation": "plus/validation-00000-of-00001.parquet",
            "test": "plus/test-00000-of-00001.parquet",
        },
        "text_column": "text",
        "label_column": "intent",
        "licence": "CC-BY-3.0",
        "n_test": 5500,
        "n_labels": 151,
    },
}


def label_names(table, column):
    """The `ClassLabel` names, in index order, out of the parquet's own metadata.

    Raises rather than falling back: a label set guessed from memory is how a
    whole run comes out shifted by one and still looks plausible."""
    blob = (table.schema.metadata or {}).get(b"huggingface")
    if blob is None:
        raise ValueError("no `huggingface` schema metadata in this parquet file: "
                         "the class names cannot be read from it")
    info = json.loads(blob.decode())
    features = info.get("info", {}).get("features") or info.get("features") or {}
    spec = features.get(column)
    if not isinstance(spec, dict) or "names" not in spec:
        raise ValueError(f"column {column!r} is not a ClassLabel in this file")
    return list(spec["names"])


def rows_from_table(table, split, text_column, label_column):
    """-> [{"id", "split", "text", "label"}], label as its *name*.

    The id is positional in the file and is assigned here, before anything
    shuffles or limits, because it is the resume key and the run-spec key."""
    names = label_names(table, label_column)
    texts = table.column(text_column).to_pylist()
    labels = table.column(label_column).to_pylist()
    return [{"id": f"{split}-{i}", "split": split, "text": text, "label": names[label]}
            for i, (text, label) in enumerate(zip(texts, labels))]


def shuffled(rows, seed):
    """One fixed permutation, shared by every model and every arm.

    See the module docstring: this is what makes `--limit N` a sample instead of
    N rows of whichever label happens to sort first."""
    out = list(rows)
    random.Random(seed).shuffle(out)
    return out


def stratified_split(rows, val_frac, seed):
    """-> (train, val), carved out of *train* and never out of test.

    Every label contributes `round(n * val_frac)` rows but never fewer than one,
    so the temperature fit sees every class. Same shape as `banking77/data.py`,
    copied rather than imported: one experiment, one directory."""
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


def _read(name, split, cache_dir):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    spec = DATASETS[name]
    if split not in spec["files"]:
        raise SystemExit(f"{name} has no {split!r} split; it has {sorted(spec['files'])}")
    path = hf_hub_download(repo_id=spec["repo"], filename=spec["files"][split],
                           repo_type="dataset", cache_dir=str(cache_dir))
    return pq.read_table(path)


def load(name, split, cache_dir):
    """The split, downloaded once and cached, in file order. Unshuffled: the
    caller shuffles, so the seed is the caller's and is recorded in the spec."""
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    table = _read(name, split, cache_dir)
    return rows_from_table(table, split, spec["text_column"], spec["label_column"])


def options(name, cache_dir):
    """Every label the dataset defines, in the file's own index order.

    Taken from the test split's metadata rather than from the labels that happen
    to appear in a sample: the option list must be the same 4 or 151 strings for
    every example, or the models are not answering the same question."""
    spec = DATASETS[name]
    return label_names(_read(name, "test", cache_dir), spec["label_column"])

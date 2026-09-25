"""Two classification tasks, four options and one hundred and fifty-one.

Both are public, labelled, and loaded by this experiment itself. **This
module imports nothing from another experiment.** It used to pull its 77-option
task out of `banking77/`, which was a mistake: that experiment deliberately
*varies* its option texts as part of its method, so this experiment's task
silently changed whenever that one ran an arm. One experiment, one directory,
self-contained -- the same choice `rerank/` makes when it copies from `arena/`
rather than importing.

| task | options | rows | licence |
| --- | --- | --- | --- |
| `ag_news` | 4 | 7,600 test | **unknown** -- the hub says so; recorded as unknown |
| `clinc150` | 151 | 5,500 test | CC-BY-3.0 |

The pair brackets the option count as widely as public labelled data allows, and
that is the axis everything here is measured along: options are free for a
decision model and are charged for on every LLM call, so a 151-option list is
where the difference should show.

**The option texts are mechanical.** The key is the dataset's own label string,
so the gold answer is the label and nothing is mapped. The description is the
label made readable -- underscores to spaces for CLINC's `oil_change_how`, and
nothing at all for AG News, whose four labels are already words. Writing prose
descriptions would be the one place this experiment could put a thumb on the
scale, so it does not.

The old `highway` task is gone. It had no ground truth, so it could only produce
model-to-model agreement, and 4-against-151 brackets the option count far better
than 5 did.
"""
import json
import random

from . import spec

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


def load_split(name, split, cache_dir):
    """The split, downloaded once and cached, in file order. Unshuffled: the
    caller shuffles, so the seed is the caller's and is recorded in the spec."""
    if name not in DATASETS:
        raise SystemExit(f"unknown dataset {name!r}; known: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    table = _read(name, split, cache_dir)
    return rows_from_table(table, split, spec["text_column"], spec["label_column"])


def options_of(name, cache_dir):
    """Every label the dataset defines, in the file's own index order.

    Taken from the test split's metadata rather than from the labels that happen
    to appear in a sample: the option list must be the same 4 or 151 strings for
    every example, or the models are not answering the same question."""
    spec = DATASETS[name]
    return label_names(_read(name, "test", cache_dir), spec["label_column"])


NAMES = ("ag_news", "clinc150")
CACHE = "typed_decisions/.cache"

# One instruction line per task, verbatim on every call to every model. It is in
# the spec and therefore in the hash, so rewording one is a new experiment and
# the report refuses to mix the two.
INSTRUCTIONS = {
    "ag_news": "Classify the news article below into exactly one of the topics listed.",
    "clinc150": "Classify the user's utterance below into exactly one of the intents listed. "
                "Answer 'oos' if it matches none of them.",
}

# How many of the leading examples the position-bias arms cover. A subset on
# purpose -- at 151 options that arm is six extra calls per example per model --
# and the size is stated everywhere the numbers are reported.
BIAS_SUBSET = 40
N_RANDOM_ORDERS = 3

# Carved out of *train*, never out of test, and used only to fit the one
# temperature in `stats.py`.
VALIDATION_N = 100
VALIDATION_FRACTION = 0.1


def describe(label):
    """The readable form of a dataset label. `oil_change_how` -> `oil change
    how`; a label that is already words gets no description, which `prompts.py`
    renders as the bare key -- exactly how laya renders it, so the typed and the
    text transports still show the same thing."""
    readable = label.replace("_", " ")
    return None if readable == label else readable


def option_texts(labels):
    """-> an ordered {label: description or None}, in the dataset's index order.

    The dataset's order: not alphabetical, not shuffled. It is the canonical
    order the spec pins, and every position-bias arm is a permutation of it."""
    return {label: describe(label) for label in labels}


def build(name, rows, labels, limit=0, seed=0, bias_subset=BIAS_SUBSET,
          n_random_orders=N_RANDOM_ORDERS, validation=(), source=""):
    """The task dict plus its frozen spec. Pure: no network, no disk.

    `rows` arrive in file order and are shuffled **here**, before the limit, with
    the seed that goes into the spec. CLINC's test split ships sorted by intent,
    so a limit applied to the raw order would measure two intents out of 151."""
    if name not in NAMES:
        raise SystemExit(f"unknown task {name!r}; known: {', '.join(NAMES)}")
    sampled = shuffled(rows, seed)
    cut = limit or len(sampled)
    examples, warmup_pool = sampled[:cut], sampled[cut:]
    criteria = option_texts(labels)
    dataset = DATASETS[name]
    frozen = spec.build(
        task=name, dataset=dataset["repo"], config=dataset["config"], split="test",
        seed=seed, examples=examples, instructions=INSTRUCTIONS[name],
        option_texts=criteria, bias_subset=min(bias_subset, len(examples)),
        n_random_orders=n_random_orders, validation=list(validation),
    )
    return {
        "name": name,
        "instructions": INSTRUCTIONS[name],
        "criteria": criteria,
        "examples": examples,
        "warmup_pool": warmup_pool,
        "validation": list(validation),
        "spec": frozen,
        "has_gold": True,
        "licence": dataset["licence"],
        "source": source or f"{dataset['repo']} ({dataset['config']} config), test split",
    }


def load(name, limit=0, seed=0, cache_dir=CACHE, bias_subset=BIAS_SUBSET,
         n_random_orders=N_RANDOM_ORDERS, validation_n=VALIDATION_N):
    """Downloads the splits once, then `build`.

    The option list comes from the dataset's own `ClassLabel` names, so 4 and 151
    are the files' counts rather than mine. The validation rows come out of
    **train**, stratified, and are the only thing anything is ever fitted on."""
    if name not in NAMES:
        raise SystemExit(f"unknown task {name!r}; known: {', '.join(NAMES)}")
    rows = load_split(name, "test", cache_dir)
    labels = options_of(name, cache_dir)
    validation = []
    if validation_n:
        train = load_split(name, "train", cache_dir)
        _, held_out = stratified_split(train, VALIDATION_FRACTION, seed)
        validation = shuffled(held_out, seed)[:validation_n]
    return build(name, rows, labels, limit=limit, seed=seed, bias_subset=bias_subset,
                 n_random_orders=n_random_orders, validation=validation)

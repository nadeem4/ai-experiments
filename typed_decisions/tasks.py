"""Two classification tasks, four options and one hundred and fifty-one.

Both are public, labelled, and loaded by this experiment's own `data.py`. **This
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
from . import data, spec

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
# temperature in `calibration.py`.
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
    shuffled = data.shuffled(rows, seed)
    cut = limit or len(shuffled)
    examples, warmup_pool = shuffled[:cut], shuffled[cut:]
    criteria = option_texts(labels)
    dataset = data.DATASETS[name]
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
    rows = data.load(name, "test", cache_dir)
    labels = data.options(name, cache_dir)
    validation = []
    if validation_n:
        train = data.load(name, "train", cache_dir)
        _, held_out = data.stratified_split(train, VALIDATION_FRACTION, seed)
        validation = data.shuffled(held_out, seed)[:validation_n]
    return build(name, rows, labels, limit=limit, seed=seed, bias_subset=bias_subset,
                 n_random_orders=n_random_orders, validation=validation)

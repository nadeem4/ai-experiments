# AI Experiments

Measured experiments on AI models, each one keeping the exact request and response of every call it made. Completed results are published at **[lab.codewithnk.com](https://lab.codewithnk.com)**.

Every experiment is two files: **`README.md` says what it is and how to run it, written before the run, and `RESULTS.md` is what happened, written after.** Two files rather than two sections, so a prediction cannot be quietly reworded once the answer is known and `git log` shows which came first. [`EXPERIMENT_TEMPLATE.md`](EXPERIMENT_TEMPLATE.md) is the skeleton to copy for a new one.

## The experiments

An experiment with no result link has not produced one yet.

| Experiment | Dataset | What it is trying to find out | Result | |
|---|---|---|---|---|
| `rerank` | [BEIR NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus) test<br><sub>3,633 docs, 323 queries, top-20</sub><br>[what it holds](rerank/DATASET.md) | Whether a decision model re-ranks retrieval better than BM25, and what that costs | — | [README](rerank/README.md) |
| `banking77` | [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) test<br><sub>3,080 rows, 77 intents</sub><br>[what it holds](banking77/DATASET.md) | Whether a published accuracy is explained by a shared per-option token budget, or by something else | [RESULTS](banking77/RESULTS.md) | [README](banking77/README.md) |
| `typed_decisions` | [AG News](https://huggingface.co/datasets/fancyzhx/ag_news) test, 4 labels<br>[CLINC150](https://huggingface.co/datasets/clinc/clinc_oos) `plus` test, 151 labels<br><sub>300 examples each</sub><br>[what they hold](typed_decisions/DATASET.md) | How nine models compare on accuracy, latency, cost, validity and positional robustness for one fixed-label job, at 4 options and at 151 | [RESULTS](typed_decisions/RESULTS.md) | [README](typed_decisions/README.md) |

## Running one

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12. One command runs any experiment, here or on a Kaggle GPU. It is idempotent: work already in the wire log is skipped, so a second invocation recomputes the tables in seconds rather than measuring again.

```
uv sync

uv run cli list                                  # every experiment, and how far it got
uv run cli run    <name> [--tag full] [flags]    # run it on this machine
uv run cli run    <name> --help                  # the standard flags plus that experiment's own
```

`--tag` names both `runs/<tag>/` and `results/<tag>/`. A tag that has run before reuses the settings it ran with, so re-running one means the same measurement rather than today's flag defaults.

**On a Kaggle GPU**, for the experiments that have a notebook at `<name>/kaggle/`:

```
uv run cli submit <name> --dry-run               # check the prerequisites, push nothing
uv run cli submit <name>                         # push and start
uv run cli status <name>
uv run cli fetch  <name>                         # bring results and the wire log back
```

`submit` refuses when submitting would be wrong, and reports every reason at once. The kernel clones this repository at `main`, so uncommitted work, unpushed commits and the wrong branch each produce a run that looks entirely normal while measuring code that is not in front of you. Two things it cannot check — whether the notebook's accelerator is on and whether its secret is attached — it prints as steps with the URL.

## Where the output goes

Each experiment keeps everything it produces inside itself:

| | |
|---|---|
| `<name>/results/<tag>/` | committed: `summary.json`, CSV tables, `figures/` |
| `<name>/runs/<tag>/` | gitignored: the raw wire log and the frozen run config |

The split is what makes "the results reproduce byte-identically" a checkable claim rather than an assertion: the log is the evidence, the results are the claim, and re-running the report derives one from the other. Each `RESULTS.md` names its own files and what they mean.

```
uv run pytest
```

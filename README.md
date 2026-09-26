# AI Experiments

Measured experiments on AI models, each one carrying the raw wire it was measured from. **Published at [lab.codewithnk.com](https://lab.codewithnk.com).** A sibling repo runs the [Decision Arena](https://arena.codewithnk.com), where the same models play games.

Every experiment is two files: **`README.md` says what it is and how to run it, written before the run, and `RESULTS.md` is what happened, written after.** Two files rather than two sections, so a prediction cannot be quietly reworded once the answer is known and `git log` shows which came first. [`EXPERIMENT_TEMPLATE.md`](EXPERIMENT_TEMPLATE.md) is the skeleton to copy for a new one.

## Running one

One command runs any experiment, here or on a Kaggle GPU. It is idempotent: work already in the wire log is skipped, so a second invocation recomputes the tables in seconds rather than measuring again.

```
uv sync

uv run cli list                                  # every experiment, its status, its results
uv run cli run    <name> [--tag full] [flags]    # run it on this machine
uv run cli run    <name> --help                  # the standard flags plus that experiment's own
```

`--tag` names both `runs/<tag>/` and `results/<tag>/`. A tag that has run before reuses the settings it ran with, so re-running one means the same measurement rather than today's flag defaults.

**On a Kaggle GPU**, for the experiments that have a notebook (`<name>/kaggle/`):

```
uv run cli submit <name> --dry-run               # check the prerequisites, push nothing
uv run cli submit <name>                         # push and start
uv run cli status <name>
uv run cli fetch  <name>                         # bring results and the wire log back
```

`submit` refuses when submitting would be wrong and reports every reason at once. The kernel clones this repository at `main`, so uncommitted work, unpushed commits and the wrong branch each produce a run that looks entirely normal while measuring code that is not in front of you. Two things it cannot check — whether the notebook's accelerator is on and whether its secret is attached — it prints as steps with the URL.

Each experiment keeps everything it produces inside itself:

| | |
|---|---|
| `<name>/results/<tag>/` | committed: `summary.json`, CSV tables, `figures/` |
| `<name>/runs/<tag>/` | gitignored: the raw wire log and the frozen run config |

The split is what makes "the results reproduce byte-identically" a checkable claim: the log is the evidence and the results are the claim, and re-running the report derives one from the other.

## The experiments

Two are finished. The third has been measured once and is being measured again.

### 1. Re-ranking retrieval with a decision model, `rerank/`

**`re-running`.** [What it is](rerank/README.md) · [Results](rerank/RESULTS.md)

*Does putting a decision model in front of a retrieval ranking make the ranking better, and what does it cost?*

**Measured once and being measured again.** The first run found Jev moving nDCG@10 **+0.0347** against the BM25 floor and a MiniLM cross-encoder **+0.0204**, with both Laya checkpoints re-ranking **worse than doing nothing**. It was measured on CPU over the Vercel AI Gateway, whose key has since been deleted, so it cannot be reproduced; its results files were removed and the whole thing is being re-measured on a Kaggle GPU, where `laya` runs float16 rather than float32. Those numbers stand as the account of the first run and nothing else.

### 2. The Banking77 token budget, `banking77/`

**`complete`.** [What it is](banking77/README.md) · [Results](banking77/RESULTS.md)

*Is Laya's published 0.425 on Banking77 explained by the shared `head_max_len` option budget?*

Inference only, nothing trained. **Finding: no.** Giving every option all the room it needs removes every truncation collision and buys **4.8 points**, 0.4403 to 0.4883 across all 3,080 test rows (McNemar p = 2.7e-12) — about a ninth of the 43-point gap to Jev's published 0.870, and then it stops buying anything: 384, 512 and 768 are identical to every decimal. Meanwhile accuracy falls from 0.743 to 0.440 as options go from ten to seventy-seven **with the budget untouched**. The documented coarse-to-fine workaround scored **below doing nothing**, 0.344 against 0.440.

### 3. Many models, one job, `typed_decisions/`

**`complete`.** [What it is](typed_decisions/README.md) · [Results](typed_decisions/RESULTS.md)

*For one job — assign a piece of text to exactly one of a fixed list of labels — how do a decision model and a wide slate of hosted LLMs compare on accuracy, latency, cost, validity, positional robustness and whether they return a probability at all, at 4 options and at 151?*

Nine models — GLM, Phi, Gemma, DeepSeek, Llama, Qwen, GPT, Jev and Laya — on **AG News** (4 labels) and **CLINC150** (151 labels), with ids resolved from OpenRouter's live catalogue rather than written down.

Two things make it different from the two above. **Every model reads its inputs off one frozen run spec** — the example ids, the instruction text, the option texts and every option ordering — hashed with SHA-256, with that hash on every stored call, so a model added next month is comparable against today's numbers without re-running anything, and the report **refuses** to mix records from two specs. And **position bias is measured causally**: the same example under three seeded orderings for a flip rate, plus the correct option placed deliberately first, middle and last for accuracy by position. The prediction about that was committed before the run, and `git log` shows it.

**Finding.** 10,120 calls for $0.72. Jev is the **fastest model on both tasks** — 1.7x to 15.6x ahead, with the lowest tail ratio and a median that rises 8% from 4 options to 151 while `gemma`'s rises 74% — and it **leads on accuracy on neither**. It and Laya are the only two models that return a probability at all, which is a capability the other seven simply lack.

The position bias the experiment was built to measure turns up in **one model/task pair out of eighteen**: `phi` loses 22.5 points when the correct option moves to the bottom of a 151-item list, 13 discordant pairs splitting 11 to 2, p = 0.0225. Every other pair rests on nought to four discordant examples and the 40-example subset cannot resolve them. So the prediction's LLM half did not hold, and its decision-model half "held" in a run where almost nothing moved — which makes it untested rather than confirmed. Laya cannot take a 151-option list at all: 151 intent names exceed its 192-token budget, and all 300 calls fail structurally.

*Transport note:* this experiment ran over the Vercel AI Gateway before 2026-09-24 and over OpenRouter after it. The two sets of numbers are not comparable, every call record carries its `transport`, and the report will not merge them. It has also been renamed twice, from `latency/` to `decision_cost/` to `typed_decisions/`; `git log --follow` traces it.

## What a decision model is

It does not write text. You hand it the situation in words and the options you will accept, and it hands back a number for every option: a probability, or a position on a scale you defined. There is nothing to parse, and it cannot answer with something that is not on the list.

Two of them appear across these experiments. **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** is TypeSafe AI's hosted System One model, closed weights, reached over HTTP. **[Laya](https://github.com/NandhaKishorM/laya)** is Convai's open-weights alternative, 421M parameters, downloaded and run on your own machine. Both answer the same typed questions, so the same request can be sent to either.

## The site

`site/` publishes all three experiments at [lab.codewithnk.com](https://lab.codewithnk.com). The re-ranking page is the most worked, written as an experiment rather than a results dump: the question, why it is worth asking, the prediction, the mechanism, the findings, the boundaries on them, and the conclusions.

Four hand-built SVG charts carry the argument, with no charting library:

- **the strip plot**, all 323 queries as one dot each, four rows, positioned by their nDCG@10 change against BM25, with a vertical line at zero. It shows what the four means hide: the biggest single group of queries is the one where nothing moved. It is also the navigation, because clicking a dot opens that query below.
- **the slope chart**, BM25's 20 positions on the left against the chosen re-ranker's on the right, judged-relevant passages emphasised from the qrels. Switching method animates the re-order, and honours `prefers-reduced-motion`.
- **the quantisation rug**, every distinct value Jev and the cross-encoder returned, each on its own range plus a magnified sixteenth of it, which is where Jev's two-decimal grid becomes visible.
- **the interval plot**, the four paired differences and their 95% intervals against one zero line, so "excludes zero" can be checked rather than believed.

The table, the charts and the CSV are the whole run. The browsable examples are **a curated subset of 24 queries**, since the full wire is 25,840 records of roughly 3 KB and cannot be shipped to a browser, chosen by `rerank/examples.py` to span the outcomes, including the queries where Jev's calls failed. That selection is the only judgement call in the export, so it is a pure function with tests. The page says the same thing.

`rerank/scripts/export_examples.py` writes everything the re-ranking page reads:

| File | What it is |
|---|---|
| `site/data/results.json`, `site/data/pilot.json` | the two run files, copied unchanged |
| `site/public/results.json` | the full-scale run file again, where a browser can download it |
| `site/data/per-query.json` | one row per query: BM25's own nDCG@10 and each method's difference from it. The strip plot and the CSV download are both built from this |
| `site/data/scores.json` | every distinct score Jev and the cross-encoder returned, for the rug |
| `site/data/question.json` | one recorded call, so the page shows the real question rather than a retyped one |
| `site/data/examples.json`, `site/public/examples/*.json` | the curated subset and its wire |

```
uv run python -m rerank.scripts.export_examples   # results + the curated wire -> site/
cd site && npm install && npm run build    # static export to site/out/
cd site && npm test && npm run lint        # the chart geometry and the CSV are unit tested
```

The site's own pure logic is tested in `site/lib/*.test.ts`: the strip-plot scale and stacking, the slope-chart position mapping, the rug bucketing, the CSV generation and the JSON tokeniser.

## The suite

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

```
uv sync
uv run pytest
```

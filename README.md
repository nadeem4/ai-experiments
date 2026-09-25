# AI Experiments

Measured experiments on AI models, each one carrying the raw wire it was measured from. **Published at [lab.codewithnk.com](https://lab.codewithnk.com).** A sibling repo runs the [Decision Arena](https://arena.codewithnk.com), where the same models play games.

Every experiment is two files: **`README.md` is the protocol, written before the run, and `RESULTS.md` is what happened, written after.** Two files rather than two sections, so a prediction cannot be quietly reworded once the answer is known and `git log` shows which came first. [`EXPERIMENT_TEMPLATE.md`](EXPERIMENT_TEMPLATE.md) is the skeleton to copy for a new one.

## The experiments

One is finished. Two are half-built: their harnesses are complete and a pilot has run, but the full run has not.

### 1. Re-ranking retrieval with a decision model, `rerank/`

**`complete`.** [Protocol](rerank/README.md) · [Results](rerank/RESULTS.md)

*Does putting a decision model in front of a retrieval ranking make the ranking better, and what does it cost?*

**Finding.** Over all 323 BEIR NFCorpus test queries, Jev moved nDCG@10 by **+0.0347** against the BM25 floor and a MiniLM cross-encoder by **+0.0204**, while both Laya checkpoints re-ranked **worse than doing nothing**. All four paired intervals exclude zero. The whole Jev pass cost $0.1533.

### 2. The Banking77 token budget, `banking77/`

**`piloted`.** [Protocol](banking77/README.md) · [Results](banking77/RESULTS.md)

*Is Laya's published 0.425 on Banking77 explained by the shared `head_max_len` option budget?*

Inference only, nothing trained. **Finding so far.** A 200-example pilot of arm A reproduces the published number, 0.435 with a 95% interval of 0.365 to 0.505, and the truncation diagnostics show five pairs of labels collapsing onto the same string at the default budget. **The budget sweep itself has not run**, so the question is not answered yet.

### 3. Many models, one job, `typed_decisions/`

**`piloted`.** [Protocol](typed_decisions/README.md) · [Results](typed_decisions/RESULTS.md)

*For one job — assign a piece of text to exactly one of a fixed list of labels — how do a decision model and a wide slate of hosted LLMs compare on accuracy, latency, cost, validity, positional robustness and whether they return a probability at all, at 4 options and at 151?*

Nine models — GLM, Phi, Gemma, DeepSeek, Llama, Qwen, GPT, Jev and Laya — on **AG News** (4 labels) and **CLINC150** (151 labels), with ids resolved from OpenRouter's live catalogue rather than written down.

Two things make it different from the two above. **Every model reads its inputs off one frozen run spec** — the example ids, the instruction text, the option texts and every option ordering — hashed with SHA-256, with that hash on every stored call, so a model added next month is comparable against today's numbers without re-running anything, and the report **refuses** to mix records from two specs. And **position bias is measured causally**: the same example under three seeded orderings for a flip rate, plus the correct option placed deliberately first, middle and last for accuracy by position. The protocol records a falsifiable prediction about that before the run.

**Finding so far.** A 358-call pilot confirms every transport on both tasks for $0.0293, and confirms two things the protocol got wrong: **Jev is not free on options** (363 to 2,264 input tokens and 47 to 1,301 *output* tokens between 4 options and 151 — a shallower slope from a higher floor, not a flat one), and **Laya cannot take a 151-option list at all**, since 151 intent names exceed its 192-token option budget. At 151 options the option list is 90–96% of what an LLM is billed for. **Position bias is not yet measured**: the pilot's bias subset was 2 examples. The full run is priced at $0.60–$1.00 and has not been launched.

*Transport note:* this experiment ran over the Vercel AI Gateway before 2026-09-24 and over OpenRouter after it. The two sets of numbers are not comparable, every call record carries its `transport`, and the report will not merge them. It has also been renamed twice, from `latency/` to `decision_cost/` to `typed_decisions/`; `git log --follow` traces it.

## What a decision model is

It does not write text. You hand it the situation in words and the options you will accept, and it hands back a number for every option: a probability, or a position on a scale you defined. There is nothing to parse, and it cannot answer with something that is not on the list.

Two of them appear across these experiments. **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** is TypeSafe AI's hosted System One model, closed weights, reached over HTTP. **[Laya](https://github.com/NandhaKishorM/laya)** is Convai's open-weights alternative, 421M parameters, downloaded and run on your own machine. Both answer the same typed questions, so the same request can be sent to either.

## The site

`site/` publishes the re-ranking experiment at [lab.codewithnk.com](https://lab.codewithnk.com), written as an experiment rather than a results dump: the question, why it is worth asking, the prediction, the mechanism, the findings, the boundaries on them, and the conclusions.

Four hand-built SVG charts carry the argument, with no charting library:

- **the strip plot**, all 323 queries as one dot each, four rows, positioned by their nDCG@10 change against BM25, with a vertical line at zero. It shows what the four means hide: the biggest single group of queries is the one where nothing moved. It is also the navigation, because clicking a dot opens that query below.
- **the slope chart**, BM25's 20 positions on the left against the chosen re-ranker's on the right, judged-relevant passages emphasised from the qrels. Switching method animates the re-order, and honours `prefers-reduced-motion`.
- **the quantisation rug**, every distinct value Jev and the cross-encoder returned, each on its own range plus a magnified sixteenth of it, which is where Jev's two-decimal grid becomes visible.
- **the interval plot**, the four paired differences and their 95% intervals against one zero line, so "excludes zero" can be checked rather than believed.

The table, the charts and the CSV are the whole run. The browsable examples are **a curated subset of 24 queries**, since the full wire is 25,840 records of roughly 3 KB and cannot be shipped to a browser, chosen by `rerank/examples.py` to span the outcomes, including the queries where Jev's calls failed. That selection is the only judgement call in the export, so it is a pure function with tests. The page says the same thing.

`scripts/export_examples.py` writes everything the site reads:

| File | What it is |
|---|---|
| `site/data/results.json`, `site/data/pilot.json` | the two run files, copied unchanged |
| `site/public/results.json` | the full-scale run file again, where a browser can download it |
| `site/data/per-query.json` | one row per query: BM25's own nDCG@10 and each method's difference from it. The strip plot and the CSV download are both built from this |
| `site/data/scores.json` | every distinct score Jev and the cross-encoder returned, for the rug |
| `site/data/question.json` | one recorded call, so the page shows the real question rather than a retyped one |
| `site/data/examples.json`, `site/public/examples/*.json` | the curated subset and its wire |

```
uv run python scripts/export_examples.py   # results + the curated wire -> site/
cd site && npm install && npm run build    # static export to site/out/
cd site && npm test && npm run lint        # the chart geometry and the CSV are unit tested
```

The site's own pure logic is tested in `site/lib/*.test.ts`: the strip-plot scale and stacking, the slope-chart position mapping, the rug bucketing, the CSV generation and the JSON tokeniser.

## The suite

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12. Each experiment's own commands are in its protocol file.

```
uv sync
uv run pytest
```

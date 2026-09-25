# What does one typed decision cost?

**Status:** `piloted`

Results so far: [RESULTS.md](RESULTS.md). This file is the protocol.

## The question

For the same decision, what does one answer cost in latency, tokens, money and validity, asked of a decision model against one LLM per family?

## Motivation

A decision model answers a typed question in one forward pass and hands back a probability per option. An LLM answers by generating text that has to be parsed back into a decision. Everything hosted goes through the **Vercel AI Gateway**. Laya runs locally on this machine's CPU and is grouped separately in every table, because that number is not comparable to a hosted call.

What changes depending on the answer: it decides whether a typed decision is worth routing to a decision model at all. If one decision costs the same either way, the LLM wins on flexibility and nobody should add a second kind of model to their stack. If the option list is what an LLM is billed for, then the wider the label set, the stronger the case for a model that does not carry the options in its prompt, and the `banking77/` and `rerank/` experiments are asking the right question.

## What we expect

The prediction and where it is recorded, all of it in the harness that produced the pilot:

- **Options are free for a decision model and expensive for an LLM.** Recorded in the module docstring of `latency/tasks.py`: *"The point of the pair is that options are free for a decision model and are charged for on every LLM call. So the same models answer the same kind of question twice, once with five options and once with seventy-seven, and the report shows how the cost of one decision scales with the length of the option list."* The same sentence heads the design section of this protocol, and commit `36047b5` states it again.
- **There is a parsing tax, and it should show up where the option list is long.** Recorded in the module docstring of `latency/parse.py`: *"a model that is fast and returns prose you cannot parse has not done the job"*, which is why validity is a first-class result and why nothing is repaired or re-asked.
- **The account, not the model, is expected to dominate wall clock.** Recorded in commit `36047b5`: *"Latency is split across two clocks, the successful attempt, and wall clock including retry backoff, so a rate limit is never charged to a model's speed and never disappears either."*

**What would falsify the first one:** the measured option-token overhead being a small share of the prompt at 77 options, or the per-decision cost of a decision model and an LLM staying within noise of each other as the option count grows. The two-task design exists so that this is a measurement rather than an argument: the same models answer the same kind of question at 5 options and at 77.

**Git evidence, stated plainly.** `latency/` arrived in the repository as one commit, `36047b5` (2026-09-24T16:22:58-05:00), which carries the harness, the docstrings quoted above, the pilot's run artifacts and the first version of this write-up together. So `git log` proves those records predate this documentation change, and it does **not**, on its own, prove each docstring was typed before the pilot ran. What it shows is that the quoted sentences live in the modules the pilot was executed through: `tasks.py` built the examples and `parse.py` classified every answer. The run's own `config.json` is stamped `2026-09-24T16:22:15`, 43 seconds before the commit.

## Data

Two tasks, five options and seventy-seven, both built from data that already exists in this repo or next to it. Same examples for every model within a task, one fixed seed (0), shared permutation.

| task | options | where it comes from | ground truth |
| --- | --- | --- | --- |
| `highway` | 5 | the arena's recorded highway-env episodes (`arena/runs/highway/jev/seed-*.json`), the states and the question, word for word including all five option descriptions | **none.** The game ships only `idle` and `random` baselines, no reference policy, so the report gives model-to-model **agreement**, not accuracy |
| `banking77` | 77 | the existing [`banking77/`](../banking77/README.md) harness: its loader, its frozen option texts and its seeded shuffle are imported, not reimplemented, so the examples are the same rows in the same order | the benchmark label |

Episodes are recorded in step order, so the examples are shuffled once with a fixed seed shared by every model, which makes a `--limit` a sample rather than a prefix.

**Licence:** not recorded in this repository for either source, and this protocol does not establish it. The highway episodes are this project's own run artifacts; BANKING77 comes from the PolyAI repository named in [`banking77/README.md`](../banking77/README.md).

## Method

```mermaid
%%{init: {'theme': 'neutral'}}%%
flowchart LR
    G[gateway catalog] --> P[probe: can this account call it?]
    P --> M[one model per family]
    T[task: instructions + options + state] --> D[decision model: typed question]
    T --> L[LLM: rendered prompt + JSON schema]
    D --> A[choice, by construction]
    L --> C[completion] --> V[parse: valid / unparseable / not an option / refusal]
    A --> R[calls.jsonl: request, response, usage]
    V --> R
```

**The arms** are six models, `gpt`, `claude`, `gemini`, `gemma`, `jev`, `laya`, each answering both tasks.

**What varies:** the model, and the option count between the two tasks.

**What is held fixed:** the instruction text and the option list, the same words for every model including Laya and Jev. Only the transport differs. Temperature 0 where the model accepts one. 20 warm-up calls per model per task, drawn from outside the measured window where there are examples to spare, so provider-side prompt caching cannot make a measured call look cheaper than a cold one.

**Seeds:** `seed 0`, recorded in `config.json`, for the shared permutation of examples. The bootstrap CI in `stats.py` takes its own fixed seed.

### The fairness rules

These decide whether the result is worth publishing.

1. **The LLMs get their best shot.** Every LLM call uses the gateway's structured-output mode, `response_format` with a `json_schema` whose `choice` property is a strict `enum` of the option keys. A family whose only reachable model cannot do that would run in a strict-prompt arm instead, and `config.json` records which arm each model ran in, because it changes both latency and validity.
2. **Identical semantics.** The instruction text and the option list are the same words for every model, including Laya and Jev. Only the transport differs.
3. **Temperature 0**, recorded per model; `null` for a model that does not accept one (the newer GPT-5 and Claude tiers reject it), never silently assumed.
4. **No retry-until-valid.** One call, one answer. Transport errors (429, 5xx) are retried because none of them is an answer; an unparseable *answer* is a result. A repaired answer is a different product with a different latency, and repairing quietly is how benchmarks lie.
5. **Cost comes off the gateway's own per-call `marketCost`.** Never tokens times a rate from a pricing page. The only place a price list is used is the pre-run spend estimate, and it is labelled as an estimate.
6. **The option-token overhead is measured, not counted.** For each model and task, the same prompt is sent twice, once with the option block and once without, and the provider's own reported prompt tokens are differenced.

### The instrument

- **`catalog.py`** no model id is written down anywhere in this experiment. The gateway is asked what it hosts, the catalog is filtered (language models that expose structured outputs and accept a temperature, minus `-fast` / `-pro` / `codex` / image variants) and ranked **cheapest first**, then the ranked candidates are **probed with one tiny call** until one answers. That last step is not optional: the catalog lists 390 models and this account is entitled to a fraction of them.
- **`prompts.py`** the one place the experiment could cheat, so it is mechanical. The instruction line goes in verbatim; every option key and description goes in verbatim and in order; an option with no description renders as the bare key, which is exactly how laya renders it. The only thing added is one shared `ANSWER_DIRECTIVE`. That is the transport, and it is identical for every LLM.
- **`gateway.py`** the wire. Chat completions for the LLMs, the v4 evaluation-model route for Jev. Rate limits and 5xx are retried with exponential backoff **outside the timer**: `latency_ms` covers only the attempt that succeeded, `wall_ms` covers the whole call including backoff, so queueing is never charged to a model's speed and never disappears.
- **`parse.py`** validity, as a first-class result. The answer is the `choice` field of a JSON object, spelled exactly as the option is spelled. No fuzzy matching, no repair, no second call.
- **`laya_local.py`** Laya from the weights on disk (`LAYA_PATH` honoured), on CPU.
- **`store.py`** append-and-flush JSONL, keyed by (model, task, example), which doubles as the resume point.
- **`run.py`** / **`report.py`** the task and the CLI, and the tables.

### Every call is on the wire

One line per (model, task, example) in `latency/runs/<tag>/calls.jsonl`, holding the exact request, the exact response and the provider's own usage block, plus latency on both clocks, the validity verdict and its detail, retries, and the gateway's per-call cost. `config.json` records every resolved model id, the gateway's version block for it, which arm it ran in, its temperature, the option-overhead measurement, the seed and the library versions.

## Metrics

- **Latency on two clocks.** `latency_ms` is the attempt that succeeded, `wall_ms` is the whole call including retry backoff. Both, because the first alone hides queueing and the second alone charges the account's rate limit to the model's speed. Reported as p50 and p95, nearest-rank so every latency reported is one some call actually had.
- **Input and output tokens**, from the provider's own usage block, because that is what the bill is computed from.
- **Cost per 1,000 decisions**, from the gateway's per-call `marketCost`.
- **The option-token overhead**, measured by differencing the prompt tokens of the same call with and without the option block. This is the quantity the experiment exists to measure.
- **Validity**, as one of `valid`, `unparseable`, `not_an_option`, `refusal`, `api_error`. A model that is fast and returns prose you cannot parse has not done the job.
- **Retries**, counted separately, because they are the account's behaviour rather than the model's.
- **Agreement** between models on `highway`, since that task has no ground truth, and **accuracy** on `banking77`, which has.

## Assumptions and limits

- **Free-tier model tiers.** This key is on the free tier, which refuses the newer tiers outright, so each family resolves to a generation-old model: `claude-3-haiku` is a 2024 model and is the only Claude this key can reach, and the GPT, Gemini and Gemma arms are also a generation or two behind what the catalog lists. Topping up the gateway would change both the quality and the latency numbers, and `--models gpt=<id>` exists for exactly that.
- **Free-tier rate limits** are what make wall clock several times the successful-attempt latency. That is an artifact of the account, not of the models, which is why the two clocks are reported separately and never merged.
- **Agreement is not quality.** With no reference policy on `highway`, the report gives pairwise agreement, and a model that answers with a constant will agree with everyone who agrees with the constant. The arena's `idle` baseline exists precisely because a constant is not a driver.
- **Laya runs on this machine's CPU.** There is no usable GPU, so its latency is not comparable to a hosted call and it is grouped separately everywhere.
- **Laya reports no tokens**, so no token or per-token cost figure exists for it.
- **`highway` has no ground truth.** Five-option accuracy is not measurable in this experiment at all.

## How to run it

```bash
# the pilot: 10 examples per model, 5-option task only
python -m latency.run --models all --tasks highway --limit 10 --tag pilot
python -m latency.report --tag pilot

# one arm, pinned to an exact model id (once the account can reach a better one)
python -m latency.run --models gpt=openai/gpt-5.4-mini --tasks highway --limit 50 --tag full
```

| flag | what it does |
| --- | --- |
| `--models` | `all`, or a comma-separated subset of `gpt,claude,gemini,gemma,jev,laya`; `name=model-id` pins one arm to an exact id |
| `--tasks` | `highway`, `banking77`, or both |
| `--limit` | first N examples per task (0 = all) |
| `--warmup` | warm-up calls per model per task, not recorded (default 20) |
| `--threshold` / `--yes-spend` | **cost guard.** Before any measured call, the run prices itself from the option-overhead probe's *measured* tokens and the catalog's price, prints the per-arm breakdown, and refuses to spend more than `--threshold` (default $1.00) without `--yes-spend` |
| `--probe-only` | measure the option-token overhead and print the price of the run, then stop |

The run is **resumable**: every (model, task, example) already in `calls.jsonl` is skipped, so a killed run is restarted with the same command.

**Expected wall clock and cost.** The pilot is 10 examples per model on the 5-option task. The projected cost and wall clock of the full run are derived from the pilot's measured per-call costs and are reported in [RESULTS.md](RESULTS.md), because they are an output of the pilot rather than an assumption made before it. Money is not the binding constraint at any size considered; wall clock is.

### Tests

```bash
python -m pytest latency/tests -q
```

The pure logic is test-driven: catalog ranking and family resolution, prompt rendering, validity classification, percentiles and the bootstrap CI, agreement, the spend estimate, the store's resume key, both task loaders, and the transport's retry behaviour against a fake HTTP post, including that the backoff never lands in the reported latency and that an answer is never asked for twice.

# Results: what does one typed decision cost?

Protocol: [README.md](README.md).

> **The pilot has run, the full experiment has not.** The pilot is 10 examples per model on the 5-option `highway` task, plus the option-token overhead probe, which also covers the 77-option task. The 77-option task has never been *answered* by any model here, only priced.

## The finding

At 77 options, **nine tenths of what an LLM is billed for on every call is the option list**, but the decision model is not free of it either: Jev's own input tokens roughly double from 5 options to 77, so on the 5-option task Jev and the cheapest LLM cost the same per decision and the gap at 77 options is about 1.6x, not 10x.

## The numbers

`--models all --tasks highway --limit 10 --tag pilot`, 20 warm-up calls per model, seed 0. Ten examples is not a result about which model drives better. It is a check that the transport works for each family, and a measurement of what a call costs.

### Which models this account can actually call

The gateway lists 390 models. This key is on the **free tier**, which refuses the newer tiers outright, so the cheapest-first ladder walked past them:

| family | resolved to | refused first |
| --- | --- | --- |
| GPT | `openai/gpt-4.1-nano` | `openai/gpt-6-luna` (403, free tier) |
| Claude | `anthropic/claude-3-haiku` | (none. It is the cheapest, and the *only* reachable Claude: `claude-haiku-4.5` is 403) |
| Gemini | `google/gemini-2.5-flash-lite` | (none. `gemini-3.1-flash-lite` is 403) |
| Gemma | `google/gemma-4-26b-a4b-it` | `google/gemma-4-31b-it` (403, free tier) |
| Jev | `typesafe-ai/jev` | (none) |
| Laya | local weights, CPU | (none) |

All four LLM families ran in **structured mode** (`json_schema`, strict, `enum` of the option keys) at **temperature 0**. Jev and Laya answer a typed question, so there is no arm to choose.

### Per model, 10 examples, 5 options

Hosted, via the gateway:

| model | id | p50 ms | p95 ms | wall p50 | in tok | out tok | $/1k decisions | valid | retries |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev | `typesafe-ai/jev` | **252** | **308** | 252 | 487 | 61 | 0.0204 | 100% | 0 |
| Gemini | `google/gemini-2.5-flash-lite` | 689 | 7093 | 689 | 195 | 8.1 | 0.0205 | 90% | 9 |
| Claude | `anthropic/claude-3-haiku` | 723 | 1067 | 854 | 637 | 39.0 | 0.1873 | 90% | 18 |
| GPT | `openai/gpt-4.1-nano` | 924 | 1245 | 4190 | 212 | 7.4 | 0.0242 | 100% | 20 |
| Gemma | `google/gemma-4-26b-a4b-it` | 951 | 2450 | 980 | 197 | 8.0 | 0.0303 | 100% | 7 |

Local, CPU, **not comparable to the rows above**, no network in it and no GPU on this machine:

| model | p50 ms | p95 ms | in tok | out tok | $/1k | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya | 368 | 473 | not reported | not reported | 0.0000 | 100% |

**The two clocks are the point of the table.** GPT's median *successful attempt* took 924 ms; its median call took 4,190 ms, because the free tier rate-limited it on 6 of 10 calls. Neither number alone is honest, so both are reported.

**Nothing failed to parse.** Every validity failure in the pilot, one Claude call and one Gemini call, was a 429 that survived five retries, recorded as `api_error`. In structured mode, on a 5-option list, all four LLMs returned a legal option every time they returned anything at all.

### What the option list costs

Measured, not counted: the same prompt twice, with and without the option block (and with the `enum` taken out of the schema but the schema left in, so that structured-output scaffolding is not charged to the options), differencing the provider's own reported prompt tokens.

| model | 5 options | share of prompt | 77 options | share of prompt |
| --- | ---: | ---: | ---: | ---: |
| `openai/gpt-4.1-nano` | 69 of 212 | 33% | 692 of 757 | **91%** |
| `anthropic/claude-3-haiku` | 84 of 637 | 13% | 785 of 1246 | 63% |
| `google/gemini-2.5-flash-lite` | 69 of 203 | 34% | 610 of 653 | **93%** |
| `google/gemma-4-26b-a4b-it` | 55 of 198 | 28% | 345 of 397 | 87% |
| Jev (whole request) | 487 total | (not separable) | 943 total | (not separable) |
| Laya | reports no tokens | (none) | reports no tokens | (none) |

At 77 options, **nine tenths of what an LLM is billed for on every single call is the option list**, the same 77 strings re-sent for every customer message. That is the structural point of the experiment.

### What the full run would cost

Projected from the measured per-call costs above, both tasks, all six arms:

| examples per model per task | highway | banking77 | total | worst-case 95% CI half-width |
| ---: | ---: | ---: | ---: | ---: |
| 200 | $0.06 | $0.13 | $0.20 | ±6.9 pp |
| **300** | **$0.09** | **$0.20** | **$0.29** | **±5.7 pp** |
| 500 | $0.15 | $0.32 | $0.46 | ±4.4 pp |
| 1000 | $0.29 | $0.62 | $0.91 | ±3.1 pp |

**300 is the recommendation.** It buys a ±5.7 pp interval on BANKING77 accuracy, enough to separate models that differ by more than about 12 points, which is the size of the gaps this comparison is looking for, and roughly 300 timed calls per model, which is enough for a p95 that is not one unlucky request. It is not enough to split two models that land within a few points of each other; that needs 1000, and 1000 costs about $0.91, still under the $1 guard.

Money is not the binding constraint. **Wall clock is**: at the pilot's observed rates the six arms together take ~25 s per example on the 5-option task, almost all of it 429 backoff, so 300 examples across both tasks is on the order of **5 to 6 hours serial**. Paid credits would cut that several-fold and unlock current models at the same time.

Raw numbers: `latency/runs/pilot/calls.jsonl` and `latency/runs/pilot/config.json`, neither committed.

## Did the prediction hold?

- **Held, and then some: options are what an LLM pays for.** The measured overhead is 87% to 93% of the prompt at 77 options for three of the four families, and 63% for the fourth. `latency/tasks.py` said the cost of one decision scales with the length of the option list, and it does.
- **Did not hold as stated: options are not free for a decision model.** Jev's own input tokens roughly double, 487 to 943, going from 5 options to 77. They are an order of magnitude cheaper per token, but "free" was the wrong word. On the 5-option task Jev and the cheapest LLM cost the *same* per decision, $0.0204 against $0.0205 per thousand. The gap only opens at 77 options, and it is about 1.6x, not 10x.
- **Not tested: the parsing tax.** `latency/parse.py` was built on the expectation that a model can be fast and still return something you cannot parse. On a 5-option list in structured mode it did not show up at all: zero parse failures across four families. The 77-option task, where the enum is long, is the one that should test it, and it has not been run.
- **Held: the account dominates wall clock.** 54 retries across 50 hosted calls, and wall clock 5 to 30 times the successful-attempt latency. Splitting the two clocks was the right call and neither number alone would have been honest.

## What surprised us

- **Jev's input tokens roughly double with the option count.** The design section called options free for a decision model. They are not; they are cheap. Nothing in the protocol predicted the size of that.
- **Gemma agreed with Jev on 10 of 10, by answering `IDLE` every time.** Agreement without a reference policy is exactly as informative as the arena's `idle` baseline, which is why that baseline exists.
- **The cost ordering at 5 options is a tie, not a win.** $0.0204 against $0.0205 per thousand decisions between Jev and the cheapest LLM was not what a "decision models are cheaper" story would predict.

## What this does not answer

- **The experiment itself.** The full run has never been launched. Everything above is 10 examples per model on one task, plus a token probe.
- **Which model decides better.** Ten examples on a task with no ground truth is not a quality measurement, and `highway` has no reference policy at any sample size.
- **Anything about the 77-option task as a task.** It has been priced, not answered. No model has produced a BANKING77 decision in this harness.
- **What current models cost.** Every hosted arm here is a generation or two behind what the catalog lists, because the key is on the free tier.
- **What the latencies would be on a paid account.** 54 retries across 50 calls means the wall clock figures measure the account's rate limit as much as the model.
- **Laya's token cost**, because Laya reports no tokens, and Laya's latency against a hosted call, because it ran on this machine's CPU with no network in it.

## Provenance

| | |
|---|---|
| GPT | `openai/gpt-4.1-nano`, v1 chat completions, `json_schema` (strict), temperature 0. Released `1744588800`, context 1,047,576 |
| Claude | `anthropic/claude-3-haiku`, v1 chat completions, `json_schema` (strict), temperature 0. Released `1710288000`, context 200,000 |
| Gemini | `google/gemini-2.5-flash-lite`, v1 chat completions, `json_schema` (strict), temperature 0. Released `1750118400`, context 1,048,576 |
| Gemma | `google/gemma-4-26b-a4b-it`, v1 chat completions, `json_schema` (strict), temperature 0. Released `1775088000`, context 262,144 |
| Jev | `typesafe-ai/jev`, v4 evaluation-model route, typed. Released `1789430400`, context 32,000 |
| Laya | local weights at `C:\projects\jev_demo\arena\models\laya`, CPU, `head_max_len` 192, `max_len` 512 |
| Gateway | Vercel AI Gateway, `https://ai-gateway.vercel.sh/v1/models` for the catalog. Free-tier key |
| Seed | 0; 20 warm-up calls per model per task; `max_tokens` 64 |
| Library versions | numpy 2.5.3, laya 0.3.7, Python 3.12.14 |
| Hardware | Windows-11-10.0.26200-SP0. Laya on CPU, no usable GPU |
| Run written at | `2026-09-24T16:22:15` (`latency/runs/pilot/config.json`) |
| Commit | `36047b5`, which is where the harness and this pilot landed. The run itself records no commit |
| Artifacts | `latency/runs/pilot/calls.jsonl`, `latency/runs/pilot/config.json`, neither committed |

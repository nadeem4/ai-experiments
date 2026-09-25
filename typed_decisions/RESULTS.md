# Results: what does one typed decision cost?

Protocol: [README.md](README.md).

> **Two runs are described in this file and they are not comparable.** The experiment moved from the Vercel AI Gateway to OpenRouter on 2026-09-24 (protocol amendment of the same date). Different provider, different routes, different models. The current numbers are the OpenRouter pilot below; the Vercel pilot is summarised at the bottom as history and is never averaged with them. Every call record carries a `transport` field and the report refuses to merge the two.

> **The pilot has run, the full experiment has not.** The OpenRouter pilot is 10 examples per model on the 5-option `highway` task, 5 of them asked a second time, plus the option-token overhead probe, which also covers the 77-option task. **The full run was not launched: it prices at $3.48, over the $3 ceiling set for it.** The 77-option task has still never been *answered* by any model here, only priced.

## The finding

On the current generation, **the decision model is the fastest arm and among the cheapest, but the headline is the spread, not the win**: Jev answers a 5-option decision at a 224 ms median for $0.0204 per thousand, while the current-generation Claude and Gemini cost **116x and 68x more per decision** — $2.36 and $1.38 per thousand — because both are reasoning models billing thinking tokens on a one-word answer. Gemma, a small open-weights model, matches Jev on price to within half a percent and beats it on nothing else. Every arm returned a legal option every time.

## The numbers

`--models all --tasks highway --limit 10 --warmup 3 --repeat 5 --tag or-pilot`, seed 0. Ten examples is not a result about which model drives better. It is a check that each transport works and a measurement of what a call costs.

### Which models this account can actually call

OpenRouter lists 460 models. The ladder is current generation first, cheapest tier within it, then a live probe. **No family fell back to an older generation: every LLM arm is a current release, and nothing was refused.**

| family | resolved to | generation | refused above it | temperature | reasoning effort |
| --- | --- | --- | --- | --- | --- |
| GPT | `openai/gpt-6-luna` | 2026-09 | (none) | not accepted -> `null` | `minimal` |
| Claude | `anthropic/claude-opus-5.5` | 2026-09 | (none) | 0 | `minimal` |
| Gemini | `google/gemini-3.8-flash` | 2026-09 | (none) | 0 | `minimal` |
| Gemma | `google/gemma-4-26b-a4b-it` | 2026-04 | (none) | 0 | not advertised -> not sent |
| Jev | `typesafe/jev-1.13-20260917` | 2026-09 | (pinned, not ranked) | n/a (typed) | n/a |
| Laya | local weights, CPU | n/a | (none) | n/a (typed) | n/a |

**The Claude row is a flagship against everyone else's cheap tier.** There is no 2026-09 Haiku or Sonnet in the catalog, so the current-generation-first rule lands on Opus. Read its cost per decision as "the newest Claude this account can reach", not as "Claude is expensive". Gemma's newest release is 2026-04; that is the family's current model, not a fallback.

All four LLM families ran in **structured mode** (`json_schema`, strict, `enum` of the option keys). Jev and Laya answer a typed question, so there is no arm to choose.

### Per model, 10 examples, 5 options

Hosted, via OpenRouter:

| model | id | p50 ms | p95 ms | tail | wall p50 | in tok | out tok | $/1k decisions | valid | retries |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Jev | `typesafe/jev-1.13-20260917` | **224** | **283** | 1.3 | 224 | 487 | 61.0 | **0.0204** | 100% | 0 |
| Gemma | `google/gemma-4-26b-a4b-it` | 604 | 3364 | **5.6** | 604 | 228 | 7.7 | **0.0202** | 100% | 0 |
| Claude | `anthropic/claude-opus-5.5` | 2420 | 2690 | 1.1 | 2420 | 518 | 14.5 | 2.3620 | 100% | 0 |
| GPT | `openai/gpt-6-luna` | 2801 | 4216 | 1.5 | 2801 | 210 | 86.5 | 0.0643 | 100% | 0 |
| Gemini | `google/gemini-3.8-flash` | 3479 | 12146 | **3.5** | 3479 | 250 | 317.9 | 1.3798 | 100% | 0 |

Local, CPU, **not comparable to the rows above**, no network in it and no GPU on this machine:

| model | p50 ms | p95 ms | tail | in tok | out tok | $/1k | valid |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Laya | 378 | 413 | 1.1 | not reported | not reported | 0.0000 | 100% |

**Zero retries and zero failures across all 90 calls**, so on this account wall clock and successful-attempt latency are identical everywhere. That is the opposite of the Vercel pilot, where free-tier rate limiting made wall clock 5 to 30 times the attempt latency. The two clocks are still reported separately, because the moment a rate limit appears they diverge again.

**Nothing failed to parse.** In structured mode, on a 5-option list, every arm returned a legal option on every call.

**Output tokens are where the money went.** Gemini emitted 318 completion tokens on average for a one-word answer and GPT 87, nearly all of it billed reasoning, against Gemma's 7.7. That, not the prompt, is why the two most expensive rows are expensive.

### Tail ratio (p95 / p50)

New metric. A model with a long tail is worse in production than a slower steady one, and a p50-only table hides it.

| model | p50 | p95 | tail ratio |
| --- | ---: | ---: | ---: |
| Gemma | 604 | 3364 | **5.6x** |
| Gemini | 3479 | 12146 | **3.5x** |
| GPT | 2801 | 4216 | 1.5x |
| Jev | 224 | 283 | 1.3x |
| Claude | 2420 | 2690 | 1.1x |
| Laya (local CPU) | 378 | 413 | 1.1x |

Gemma is the cheapest hosted arm and the second fastest at the median, and it has the worst tail of the six: one call in twenty takes 3.4 seconds. Claude is the slowest reasoning model at the median and the steadiest arm of all of them. At n=10 a p95 is the single worst call, so these ratios are indicative and not stable estimates.

### Determinism, on a repeated subset

New metric. `--repeat 5` asked the first 5 examples again, after every first pass had finished. **This is 5 examples per model, not the whole run**, and the rate is the share of them answered differently the second time.

| model | n (subset) | disagreement |
| --- | ---: | ---: |
| Jev | 5 | **0%** |
| Claude | 5 | 0% |
| Gemma | 5 | 0% |
| Laya (local CPU) | 5 | **0%** |
| GPT | 5 | **20%** |
| Gemini | 5 | **20%** |

Both decision models were exactly reproducible, as they should be: one forward pass, no sampling. Two of the four LLMs changed an answer on 1 of 5 examples. Note that **GPT was not run at temperature 0, because `gpt-6-luna` does not accept the parameter at all**, so its non-determinism is not a temperature-0 result. Gemini *was* at temperature 0 and still changed its mind. Five examples per model is far too small to rank the LLMs against each other here; it is enough to show the property is not free.

### Input tokens for one identical prompt

New reporting. Every model is sent the same example in the same words, so the spread is each model's tokenizer and protocol overhead rather than a difference in the prompt.

| model | input tokens, 5-option prompt |
| --- | ---: |
| Gemma | 198 |
| GPT | 210 |
| Gemini | 267 |
| Jev | 487 |
| Claude | 518 |

**Claude is billed 2.6x Gemma's input tokens for the same text.** Jev's 487 is not a tokenizer artifact in the same sense: it is a typed request carrying the state and all five option descriptions as structure, so it is the price of the protocol rather than of the prose.

### Cost per correct decision

New metric, test-driven, **and it has no value in this pilot**: `highway` ships no reference policy, so there is no ground truth, so accuracy is undefined and cost-per-correct with it. The report prints `n/a` rather than quietly substituting cost per call. The metric only becomes meaningful on the 77-option `banking77` task, which has a benchmark label and has not been answered by any model in this harness.

### What the option list costs

Measured, not counted: the same prompt twice, with and without the option block (and with the `enum` taken out of the schema but the schema left in, so that structured-output scaffolding is not charged to the options), differencing the provider's own reported prompt tokens.

| model | 5 options | share of prompt | 77 options | share of prompt |
| --- | ---: | ---: | ---: | ---: |
| `openai/gpt-6-luna` | 69 of 210 | 33% | 692 of 755 | **92%** |
| `anthropic/claude-opus-5.5` | 146 of 518 | 28% | 1358 of 1612 | **84%** |
| `google/gemini-3.8-flash` | 92 of 267 | 34% | 776 of 860 | **90%** |
| `google/gemma-4-26b-a4b-it` | 55 of 198 | 28% | 345 of 397 | **87%** |
| Jev (whole request) | 487 total | (not separable) | 943 total | (not separable) |
| Laya | reports no tokens | (none) | reports no tokens | (none) |

At 77 options, **84% to 92% of what an LLM is billed for on every single call is the option list**, the same 77 strings re-sent for every customer message. That reproduces the Vercel pilot's central finding on an entirely different set of models, which is the strongest thing this migration bought.

### Why the full run was not launched

Priced with `--probe-only` from the measured per-call tokens, both tasks, `--limit 300`, 20 warm-up calls per model per task:

| arm | projected |
| --- | ---: |
| `claude/banking77` | **$2.1530** |
| `claude/highway` | **$0.7526** |
| `gemini/highway` | $0.2453 |
| `gemini/banking77` | $0.2448 |
| `gpt/highway` | $0.0278 |
| `gpt/banking77` | $0.0262 |
| `jev/banking77` | $0.0127 |
| `gemma/banking77` | $0.0098 |
| `jev/highway` | $0.0065 |
| `gemma/highway` | $0.0053 |
| `laya/*` | $0.0000 |
| **TOTAL** | **$3.4840** |

**$3.48 is over the $3 ceiling set for this run, so it was stopped and reported rather than launched.** The estimate is a price-list estimate and is labelled as one, but it is well corroborated: for `claude/highway` it projects $0.7526 against $0.709 implied by the pilot's own measured `$/1k`, a 6% difference.

**One family is 84% of the bill.** `claude/*` alone is $2.91 of the $3.48, entirely because the current-generation Claude on this account is only available at the Opus tier. Pinning Claude to its cheap tier — `--models claude=anthropic/claude-haiku-4.5` — brings the projected total to roughly **$0.6**, at the cost of running one family a generation behind. That is a protocol decision, not a cleanup, so it is left open rather than taken unilaterally.

Raw numbers: `typed_decisions/runs/or-pilot/calls.jsonl` and `typed_decisions/runs/or-pilot/config.json`; the pricing pass is `typed_decisions/runs/or-full/config.json`. None committed.

## Did the prediction hold?

- **Held, on entirely new models: options are what an LLM pays for.** 84% to 92% of the billed prompt at 77 options, across four families none of which are the models that produced the same finding on the Vercel gateway.
- **Did not hold as stated: options are not free for a decision model.** Jev's own input tokens roughly double, 487 to 943, going from 5 options to 77 — the same doubling the Vercel pilot saw, since it is a property of Jev's protocol rather than of the transport.
- **Not tested: the parsing tax.** Zero parse failures across four families on a 5-option list in structured mode, again. The 77-option task is the one that should test it, and it still has not been answered.
- **Did not hold on this account: the account dominates wall clock.** Zero retries across 90 calls, so wall clock equals attempt latency everywhere. That prediction was about the Vercel free tier and does not transfer; on OpenRouter the models' own latency is what the table shows.

## What surprised us

- **The cheap open-weights model has the worst tail.** Gemma is the cheapest hosted arm and second fastest at the median, with a 5.6x tail ratio. The metric was added for exactly this case and immediately found something a p50 table would have hidden.
- **Reasoning tokens, not prompt tokens, are what make a current-generation decision expensive.** Gemini emitted 318 completion tokens for a one-word answer. The experiment was built to measure prompt-side option cost; on reasoning models the completion side is now the larger term.
- **`gpt-6-luna` does not accept a temperature at all.** The old catalog filter *required* one, which silently excluded every current-generation GPT and Claude and resolved those families to 2025 models. The migration found this only because the instruction was to report which generation each family ran at, which is a good argument for always reporting it.
- **Price, median latency and tail are three different orderings.** The two most expensive arms are not the two slowest, and the cheapest is not the fastest.

## What this does not answer

- **The experiment itself.** The full run has never been launched, on either transport. Everything above is 10 examples per model on one task, plus a token probe.
- **Which model decides better.** Ten examples on a task with no ground truth is not a quality measurement, and `highway` has no reference policy at any sample size.
- **Anything about the 77-option task as a task.** It has been priced, not answered, on either transport.
- **Cost per correct decision**, anywhere, because the only task that has been run has no ground truth.
- **Whether the LLMs are non-deterministic in general.** Five examples per model, and one of the two that moved was not at temperature 0 because it cannot be.
- **How Claude's family really compares**, because the only current-generation Claude reachable here is an Opus-tier flagship being compared against other families' cheap tiers.
- **Whether the tail ratios are stable.** At n=10 a p95 is the single worst call.
- **Laya's token cost**, because Laya reports no tokens, and Laya's latency against a hosted call, because it ran on this machine's CPU with no network in it.

## Provenance

| | |
|---|---|
| GPT | `openai/gpt-6-luna`, canonical `openai/gpt-6-luna-20260922`, `POST /api/v1/chat/completions`, `json_schema` (strict), temperature **not accepted**, `reasoning_effort` `minimal`. Released 2026-09, context 1,050,000 |
| Claude | `anthropic/claude-opus-5.5`, `POST /api/v1/chat/completions`, `json_schema` (strict), temperature 0, `reasoning_effort` `minimal`. Released 2026-09 |
| Gemini | `google/gemini-3.8-flash`, canonical `google/gemini-3.8-flash-20260902`, `POST /api/v1/chat/completions`, `json_schema` (strict), temperature 0, `reasoning_effort` `minimal`. Released 2026-09, context 1,048,576 |
| Gemma | `google/gemma-4-26b-a4b-it`, `POST /api/v1/chat/completions`, `json_schema` (strict), temperature 0, no `reasoning_effort`. Released 2026-04 |
| Jev | `typesafe/jev-1.13-20260917`, `POST /api/v1/systemone`, typed. Context 32,000. Absent from the default `/api/v1/models` listing; found under `?output_modalities=decisions` |
| Laya | local weights at `C:\projects\jev_demo\arena\models\laya`, CPU, transport `local-cpu` |
| Transport | **OpenRouter**, `https://openrouter.ai/api/v1/models` for the catalog. Free-tier key, $20 monthly cap |
| Account | $19.8785 of the $20 monthly cap remaining after the pilot and the pricing pass; $0.1215 used this month |
| Spend | pilot $0.0584 over 90 calls (60 measured + 30 repeat-pass) |
| Seed | 0; 3 warm-up calls per model per task in the pilot; `max_tokens` 512; `--repeat 5` |
| Library versions | numpy 2.5.3, laya 0.3.7, Python 3.12.14 |
| Hardware | Windows-11-10.0.26200-SP0. Laya on CPU, no usable GPU |
| Artifacts | `typed_decisions/runs/or-pilot/`, `typed_decisions/runs/or-full/config.json`, neither committed |

---

# Historical: the Vercel AI Gateway pilot (superseded)

> **These numbers are not comparable to anything above**, and nothing from them is carried into the current result. They were measured over the Vercel AI Gateway on a free-tier key that refused every current-generation model, with `max_tokens` 64 and cost read from `marketCost`. That key has since been deleted. The full original write-up is in git history at `latency/RESULTS.md` before the rename; what follows is a summary kept for one reason only, which is that the option-overhead finding reproduced across both transports.

**What it ran.** `--models all --tasks highway --limit 10 --tag pilot`, 20 warm-up calls, seed 0. Six arms, 5-option task only, 60 recorded calls. Raw records remain at `typed_decisions/runs/pilot/calls.jsonl`; they carry no `transport` field, and the report groups a record with no transport as `unknown` so it can never merge with an OpenRouter row.

**Which models it could reach.** The free tier refused every newer tier, so the cheapest-first ladder resolved each family a generation or two back: `openai/gpt-4.1-nano`, `anthropic/claude-3-haiku` (a 2024 model), `google/gemini-2.5-flash-lite`, `google/gemma-4-26b-a4b-it`, `typesafe-ai/jev`.

**What it found, and what carried over:**

- **Options are 87% to 93% of an LLM's billed prompt at 77 options** for three of four families, 63% for the fourth. This is the finding that reproduced on OpenRouter at 84% to 92%, on completely different models.
- **Options are not free for a decision model either.** Jev's input tokens went 487 to 943 from 5 options to 77. The OpenRouter run measured the identical figures, since it is a property of the protocol and not of the transport.
- **At 5 options, Jev and the cheapest LLM cost the same**, $0.0204 against $0.0205 per thousand. On OpenRouter the equivalent tie is Jev at $0.0204 against Gemma at $0.0202 — a coincidence worth noting but not a continuation of the same measurement.
- **The account, not the model, dominated wall clock**: 54 retries across 50 hosted calls and wall clock 5 to 30 times the attempt latency. This did **not** transfer to OpenRouter, which produced zero retries.
- **One model ran 689 ms p50 against 7,093 ms p95** — a 10.3x tail that the table of the day did not surface. That observation is why the tail-ratio metric now exists.

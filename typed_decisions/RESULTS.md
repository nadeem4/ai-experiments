# Results

**Status:** `complete`. 10,120 calls, nine models, two tasks, **$0.7245**.

The protocol is [README.md](README.md). Its prediction about position bias was committed in `f2893da` on **2026-09-24 22:37:30**; the full run's first call record was written on **2026-09-25 08:03**. The prediction predates the data by about nine hours and `git log -S` on the prediction line is the check.

## The finding, one sentence

Jev is the fastest model on the slate on both tasks — 1.7x to 15.6x ahead of the rest depending on the model — and one of only two that return a probability at all — and it wins on accuracy **nowhere**; meanwhile the position bias this experiment was built to measure turns out to be statistically detectable in **exactly one** of the eight models that could answer.

## What the run was

```
python -m typed_decisions.run --models all --tasks ag_news,clinc150 \
    --limit 300 --bias-subset 40 --validation 100 --tag full
```

300 examples per task in the measured `main` arm, a bias subset of **40** examples under 3 seeded random orderings and 3 controlled gold placements, and 100 validation rows carved out of **train** for the probabilistic models. 10,120 recorded calls, of which 5,400 are the measured `main` arm and 4,720 are bias and validation calls excluded from the headline tables so no example is weighted twice.

Run spec hashes: `ag_news` **ad7affdc7ebd**, `clinc150` **40e9e134db8f**. One frozen spec per task, hash-checked on read, so a model added next month is comparable against these numbers without re-running anything.

## Before the numbers: three reporter defects, found and fixed

Every number below was regenerated after `5f70067`, which fixed three places that scored a **failed call as a wrong answer**. `correct` is stored as `0` for a call that never returned, so anywhere that column was read raw, an outage counted as the model answering incorrectly.

The rule now applied everywhere: an unusable **answer** is the model failing and scores wrong, because dropping it would pay a model for returning garbage. A call that never returned is the transport and cannot be scored at all. `unparseable`, `not_an_option` and `refusal` stay wrong; `api_error` is excluded and reported in the validity column instead.

It mattered. `qwen/clinc150` read 239/300 = **0.797** against 24 API errors; over the calls it actually answered it is 239/276 = **0.866**. Seven points, and the difference between "less accurate" and "failed 8% of its calls", which are two different facts about a model. In the gold-placement table the same model went from a spread of 0.100 to **0.024**, which moved it off the falsification threshold entirely. `laya/clinc150` had been printing `0.000` at every placement — 300 structural failures rendered as a position result — and now prints `n/a`.

This is recorded here rather than quietly corrected because all three would have been published.

## The numbers

Every figure is drawn by `figures.py` from `results/full.json`, the same file the tables are printed from.

### ag_news, 4 options, n = 300

| model | id | p50 ms | p95 ms | tail | in tok | $/1k | $/correct | valid | accuracy | retries |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **jev** | `typesafe/jev-1.13-20260917` | **230** | **330** | **1.4** | 364 | 0.0153 | 0.000017 | 100% | 0.890 [0.85, 0.92] | 0 |
| glm | `z-ai/glm-5.3-flash` | 409 | 2043 | 5.0 | 114 | 0.0205 | 0.000023 | 100% | **0.910 [0.88, 0.94]** | 0 |
| phi | `microsoft/phi-4` | 482 | 972 | 2.0 | 107 | **0.0085** | **0.000010** | 100% | 0.850 [0.81, 0.89] | 0 |
| deepseek | `deepseek/deepseek-v4.1-flash` | 540 | 2927 | 5.4 | 116 | 0.0402 | 0.000046 | 100% | 0.880 [0.84, 0.91] | 0 |
| llama | `meta-llama/llama-4-scout` | 571 | 1289 | 2.3 | 109 | 0.0131 | 0.000015 | 100% | 0.857 [0.81, 0.89] | 0 |
| gemma | `google/gemma-4-26b-a4b-it` | 658 | 2259 | 3.4 | 156 | 0.0123 | 0.000014 | 100% | 0.860 [0.82, 0.90] | 0 |
| gpt | `openai/gpt-6-luna` | 1392 | 2885 | 2.1 | 138 | 0.0209 | 0.000024 | 100% | 0.867 [0.83, 0.90] | **73** |
| qwen | `qwen/qwen3.8-flash` | 2860 | 5478 | 1.9 | 168 | 0.0666 | 0.000074 | 100% | 0.900 [0.86, 0.93] | 2 |

| *local CPU, **not comparable to the rows above*** | | | | | | | | | | |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| *laya* | *local weights, CPU* | *210* | *322* | *1.5* | *n/a* | *0.0000* | *0.000000* | *100%* | ***0.913 [0.88, 0.94]*** | *0* |

### clinc150, 151 options, n = 300

| model | p50 ms | p95 ms | tail | in tok | $/1k | $/correct | valid | accuracy | retries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| **jev** | **248** | **327** | **1.3** | 2264 | 0.0951 | 0.000111 | 100% | 0.860 [0.82, 0.90] | 0 |
| glm | 425 | 2258 | 5.3 | 939 | 0.1407 | 0.000159 | 100% | 0.883 [0.85, 0.92] | 0 |
| phi | 443 | 847 | 1.9 | 928 | **0.0661** | **0.000090** | 100% | 0.737 [0.68, 0.79] | 0 |
| llama | 708 | 1309 | 1.9 | 952 | 0.0977 | 0.000122 | 100% | 0.803 [0.76, 0.85] | 13 |
| deepseek | 864 | 4673 | 5.4 | 1068 | 0.2382 | 0.000269 | 99% | 0.886 [0.85, 0.92] | 0 |
| gemma | 1144 | 3181 | 2.8 | 1311 | 0.0882 | 0.000098 | 100% | 0.897 [0.86, 0.93] | 0 |
| gpt | 1626 | 3218 | 2.0 | 1462 | 0.1941 | 0.000214 | 100% | **0.906 [0.87, 0.94]** | **66** |
| qwen | 3879 | 9640 | 2.5 | 989 | 0.2268 | 0.000262 | **92%** | 0.866 [0.82, 0.91] | 25 |
| **laya** | n/a | n/a | n/a | n/a | 0.0000 | n/a | **0%** | — | 0 |

![Cost against accuracy](results/figures/cost-accuracy.png)

*Cost per 1,000 decisions against accuracy, one point per model per task, log cost axis. There is no clean frontier: the cheapest model per correct answer on both tasks is `phi`, which is also the least accurate on CLINC150 by five points. Laya is absent because it costs a true zero and a zero has no place on a log axis; its accuracy is in the table.*

![Latency](results/figures/latency.png)

*Bar is p50, whisker reaches p95, log scale, retry backoff excluded. Two things to read: Jev's bar is the shortest on both tasks and its whisker is barely longer than its bar, and `glm` and `deepseek` carry 5x tails that a p50-only table would hide entirely.*

### Jev is fast, flat, and loses on accuracy

The speed result is unambiguous and it is the clearest thing in the run. Jev's median is **230 ms at 4 options and 248 ms at 151** — a 7.6% rise for a 38-fold increase in the answer space. Every hosted LLM pays much more for those options:

| model | ag_news p50 | clinc150 p50 | rise |
| --- | ---: | ---: | ---: |
| **jev** | 230 | 248 | **+8%** |
| glm | 409 | 425 | +4% |
| phi | 482 | 443 | −8% |
| llama | 571 | 708 | +24% |
| deepseek | 540 | 864 | +60% |
| gemma | 658 | 1144 | +74% |
| gpt | 1392 | 1626 | +17% |
| qwen | 2860 | 3879 | +36% |

**Jev is not the only model whose median holds steady**, and the table says so: `glm` rises 4% and `phi` actually falls 8%. What is true of Jev alone is the combination — it is both the flattest *and* the fastest, where `glm` and `phi` hold steady at roughly twice its median. The models that pay heavily for options are the ones in the middle of the pack.

Jev also has the **lowest tail ratio on both tasks** (1.4 and 1.3, against 5.0 and 5.4 for `glm` and `deepseek`), which in production is worth more than a median.

And it wins on accuracy nowhere. On AG News, Laya (0.913), `glm` (0.910) and `qwen` (0.900) all beat it; on CLINC150, `gpt` (0.906), `gemma` (0.897), `deepseek` (0.886) and `glm` (0.883) all do. The paired tests say how much of that is real:

**Paired against Laya, the most accurate model on AG News** — per-item differences over the examples both models answered, not two intervals:

| model | n shared | mean diff | 95% interval | |
| --- | ---: | ---: | --- | --- |
| glm | 300 | +0.003 | [−0.027, +0.033] | crosses zero |
| qwen | 299 | +0.013 | [−0.017, +0.044] | crosses zero |
| **jev** | 300 | **+0.023** | **[−0.007, +0.057]** | **crosses zero** |
| deepseek | 300 | +0.033 | [+0.000, +0.067] | crosses zero |
| gpt | 300 | +0.047 | [+0.010, +0.087] | |
| gemma | 300 | +0.053 | [+0.017, +0.090] | |
| llama | 300 | +0.057 | [+0.017, +0.097] | |
| phi | 300 | +0.063 | [+0.027, +0.100] | |

**Paired against `gpt`, the most accurate model on CLINC150:**

| model | n shared | mean diff | 95% interval | |
| --- | ---: | ---: | --- | --- |
| gemma | 299 | +0.010 | [−0.023, +0.043] | crosses zero |
| deepseek | 297 | +0.020 | [−0.010, +0.051] | crosses zero |
| glm | 299 | +0.023 | [−0.010, +0.057] | crosses zero |
| **jev** | 299 | **+0.047** | **[+0.010, +0.084]** | |
| qwen | 275 | +0.051 | [+0.022, +0.080] | |
| llama | 299 | +0.104 | [+0.060, +0.147] | |
| phi | 299 | +0.171 | [+0.117, +0.217] | |

So on AG News Jev's deficit against the best model **is not distinguishable from zero**, and on CLINC150 it **is**: 4.7 points behind `gpt`, interval clear of zero. The honest summary is that Jev trades a real but small accuracy loss for a large and reliable speed gain, and that the trade is worse at 151 options than at 4.

### Reliability is not uniform, and it is invisible in an accuracy column

| model/task | validity |
| --- | --- |
| `qwen/clinc150` | **276 valid, 24 api_error** — 8% of its calls never returned |
| `deepseek/clinc150` | 297 valid, 2 api_error, 1 unparseable |
| `gpt/clinc150` | 299 valid, 1 api_error |
| `deepseek/ag_news` | 299 valid, 1 unparseable |
| `qwen/ag_news` | 299 valid, 1 api_error |
| `laya/clinc150` | **300 api_error** — structural, see below |
| everything else | 300 valid |

`gpt` also needed **73 and 66 retries** to get its 300 answers on the two tasks, and `qwen` 2 and 25. Retry backoff is excluded from the latency figures by design, so those retries are invisible in the p50 column and real in wall clock.

### Laya cannot do CLINC150 at all

All 300 CLINC150 calls failed, verbatim from the call record:

```
ValueError: question 'decision' options exceed head_max_len=192
```

151 intent names do not fit a 192-token option budget. This is a **structural** zero, not a model getting 151-way classification wrong: the row reads 0% valid, the bias table marks it `0/240 usable` with no flip rate, and the figure refuses to draw it. Whether raising that budget recovers the accuracy is exactly what the sibling [`banking77/`](../banking77/README.md) experiment exists to answer, so it is not done here.

On AG News, where it fits, Laya is the **most accurate model in the run** at 0.913. Accuracy does not depend on the transport, so that comparison is legitimate; its latency is measured on local CPU and is grouped apart everywhere because that one is not.

### What the option list costs

| task | option tokens as a share of the billed prompt |
| --- | --- |
| ag_news, 4 options | 8% – 17% (qwen 8%, glm 10%, phi 11%, llama 11%, gemma 12%, gpt 17%) |
| clinc150, 151 options | **86% – 94%** (gemma 86%, qwen 88%, glm 93%, phi 93%, llama 93%, deepseek 93%, gpt 94%) |

At 151 options the option list is roughly nine tenths of what an LLM is billed for on **every single call** — the same list, re-sent and re-billed, for every decision.

![Option-count scaling](results/figures/option-scaling.png)

*Mean input tokens per call at 4 options against 151, log-log. Jev is not free on options either — 364 to 2,264 tokens — but its slope is shallower than the LLMs' and it starts from a much higher floor. Measured on one identical prompt, Jev is billed **3.4x** the input tokens of `deepseek` at 4 options for exactly the same text.*

One measurement is not usable: `deepseek/ag_news` reported **fewer** prompt tokens with the option list than without it, an overhead of −63. That is the provider's accounting, not a property of the options; the report flags it and no conclusion rests on it.

**Jev's output tokens scale with the option count too**, which the protocol did not anticipate: **1,301 mean output tokens** per call at 151 options. The distribution over 151 options is itself billed as output, and it is most of Jev's per-call cost at that option count.

### Position bias — the measurement this experiment exists for

![Position bias](results/figures/position-bias.png)

Bias subset **n = 40** examples per task, under 3 seeded random orderings and 3 controlled gold placements.

| model/task | opts | usable | flip | gold 1st | gold mid | gold last | spread | mean pos |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| deepseek/ag_news | 4 | 239/240 | 0% | 0.950 | 0.974 | 0.975 | 0.025 | 0.54 |
| gemma/ag_news | 4 | 240/240 | 2% | 0.925 | 0.950 | 0.925 | 0.025 | 0.54 |
| glm/ag_news | 4 | 240/240 | 8% | 0.925 | 0.950 | 0.975 | 0.050 | 0.52 |
| gpt/ag_news | 4 | 240/240 | 2% | 0.950 | 0.975 | 0.975 | 0.025 | 0.54 |
| jev/ag_news | 4 | 240/240 | **0%** | 0.975 | 0.975 | 0.975 | **0.000** | 0.54 |
| laya/ag_news | 4 | 240/240 | 2% | 0.950 | 0.975 | 0.950 | 0.025 | 0.53 |
| llama/ag_news | 4 | 238/240 | 8% | 0.925 | 0.925 | 0.950 | 0.025 | 0.54 |
| phi/ag_news | 4 | 240/240 | 5% | 0.925 | 0.925 | 0.925 | 0.000 | 0.54 |
| qwen/ag_news | 4 | 238/240 | 2% | 0.974 | 0.974 | 0.950 | 0.024 | 0.54 |
| deepseek/clinc150 | 151 | 234/240 | 8% | 0.850 | 0.825 | 0.875 | 0.050 | 0.48 |
| gemma/clinc150 | 151 | 240/240 | 15% | 0.875 | 0.875 | 0.850 | 0.025 | 0.47 |
| glm/clinc150 | 151 | 240/240 | 15% | 0.775 | 0.725 | 0.750 | 0.050 | 0.49 |
| gpt/clinc150 | 151 | 240/240 | 5% | 0.850 | 0.850 | 0.825 | 0.025 | 0.48 |
| jev/clinc150 | 151 | 240/240 | 5% | 0.825 | 0.825 | 0.850 | 0.025 | 0.50 |
| laya/clinc150 | 151 | **0/240** | n/a | n/a | n/a | n/a | n/a | n/a |
| llama/clinc150 | 151 | 240/240 | 22% | 0.725 | 0.725 | 0.700 | 0.025 | 0.42 |
| **phi/clinc150** | 151 | 240/240 | **32%** | **0.725** | 0.750 | **0.500** | **0.250** | 0.43 |
| qwen/clinc150 | 151 | 205/240 | 4% | 0.781 | 0.806 | 0.806 | 0.024 | 0.49 |

`mean pos` is where in the shown list the answer sat, 0 = always first, 1 = always last, 0.5 = no positional preference. It says what **kind** of bias a model has, not how much.

**Almost none of this spread column survives a significance test.** At n=40 per placement, a spread of 0.025 is one example. Paired McNemar (exact binomial on discordant pairs) on `gold:first` against `gold:last`, same examples in both arms:

| model/task | first | last | diff | discordant pairs | p | |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **phi/clinc150** | 0.725 | 0.500 | **+0.225** | **13** | **0.0225** | **distinguishable** |
| every other model, both tasks | | | ≤0.050 | 0 – 4 | **≥0.50** | within noise |

The smallest p-value anywhere else in the run is **0.50**, for `glm` on the 4-option task; every other pair is 1.00.

**One model out of eighteen model/task pairs shows position bias that survives a test.** Phi loses **22.5 points** when the correct option moves to the bottom of a 151-item list, while the middle placement is unharmed; its 13 discordant pairs split 11 against 2 in the predicted direction. Everything else — including both direction reversals, and including Jev's flat rows — rests on one to four discordant examples in total and is indistinguishable from zero.

What does separate the models is the **kind** of bias rather than the amount. At 151 options every model sits at or below 0.50 mean position, with `llama` (0.42) and `phi` (0.43) leaning hardest toward the top of the list and Jev sitting exactly at 0.50. That is the direction the mechanism predicts, and it is a weaker claim than the spread column was meant to support.

### Does it return a probability at all

| returns a distribution over every option | returns a label only |
| --- | --- |
| `jev`, `laya` | `glm`, `phi`, `gemma`, `deepseek`, `llama`, `qwen`, `gpt` |

Measured from the records, not declared. Seven of nine models cannot be thresholded, so "route this one to a human when the model is unsure" is not available from them at any price. This is the most durable difference in the run and it is a capability difference rather than a quality one.

### Calibration

| model/task | n test | n val | T | ECE raw | ECE scaled |
| --- | ---: | ---: | ---: | ---: | ---: |
| jev/ag_news | 300 | 100 | 1.99 | 0.0705 | **0.0768** |
| jev/clinc150 | 300 | 100 | 2.76 | 0.0674 | **0.0836** |
| laya/ag_news | 300 | 100 | 0.77 | 0.0275 | **0.0524** |

![Calibration](results/figures/calibration.png)

**Temperature scaling made calibration worse on test in all three cases.** The temperature was fitted on a validation split carved out of **train**, never on test, which is the correct procedure and also why it can lose: a temperature that helps on train-distribution rows is not guaranteed to help on test. Both Jev fits pull T above 1, meaning its raw probabilities are over-confident on validation; the correction then overshoots on test.

Laya's raw ECE of 0.0275 is the best number in this table, and it should be read against its own load-time warning, quoted verbatim from the run output:

```
laya: this checkpoint ships invalid temperatures or values outside [0.5, 5];
using choice:11+=0.10058280825614929 -> 0.5. Treat confidence from the affected
entries as uncalibrated.
```

The checkpoint says its own confidences are uncalibrated. The ECE is reported because it was measured; the warning is reported because it qualifies it.

`laya/clinc150` is absent although its 100 validation rows were collected: all 300 of its **test** calls failed structurally, so there is nothing to score the fitted temperature against. No LLM appears here at all, because none of them return a probability to calibrate.

## Did the prediction hold?

**Partly, and the run is underpowered to decide most of the rest.** Taking the protocol's clauses in order:

| clause | verdict |
| --- | --- |
| AG News, 4 options: flip under ~5% and spread within ~0.03 for every model | **Held in substance.** `glm` and `llama` flipped 8% and `glm`'s spread was 0.050, all at p = 1.0. Nothing on AG News is distinguishable from zero. |
| CLINC150, 151 options: flip **above ~20% for the LLMs** | **Did not hold.** Two of seven (llama 22%, phi 32%). `gpt` is 5% and `qwen` 4%. |
| CLINC150: accuracy varying **by more than ~0.10** across placements | **Did not hold.** One of seven (phi, 0.250). No other model exceeds 0.050. |
| CLINC150: direction favours early positions, `gold:first` > `gold:last` | **Not resolved.** Holds for five models, reverses for `deepseek` and `qwen`, but every one of those differences is one example and p = 1.0. The `mean pos` column supports the direction; the placement column cannot. |
| Jev and Laya: flip and spread at or very near zero at both option counts | **Held where testable, and it is weak evidence.** Jev is 0%/0.000 on AG News and 5%/0.025 on CLINC150. Laya is 2%/0.025 on AG News and could not answer CLINC150 at all. |

None of the four named falsifications is cleanly triggered. The one that comes closest — "Jev or Laya shows a non-trivial flip rate or spread", flagged in the protocol as the outcome that would most surprise its author — is not met: Jev's 5% flip on CLINC150 is 2 examples out of 40, the same as `gpt`, and its spread of 0.025 sits at p = 1.0.

**The honest reading is less flattering to the design than either outcome.** The prediction's decision-model half "held" in a run where almost nothing showed bias at all. Jev being flat is not evidence for the marker-scoring mechanism when `gpt`, `gemma` and `deepseek` are equally flat. At n=40 an effect must move at least **six** examples, all in the same direction, before the exact binomial can call it: six one-sided discordant pairs give p = 0.031, five give 0.063. Every model except Phi produced between **zero and four discordant pairs in total**, which is below the floor at which this arm can decide anything. It detected the one effect large enough to detect and cannot speak to the rest.

## What surprised us

1. **Position bias is not a general property of LLMs at 151 options.** The experiment was built on the assumption that long option lists induce it broadly. One model in eight shows it. That is the result, and it is the opposite of what the motivation argued.
2. **Jev's speed is flat in the option count and its accuracy is not.** +8% latency from 4 options to 151, against +60% and +74% for `deepseek` and `gemma` — while its accuracy gap against the field goes from indistinguishable-from-zero on AG News to a clear 4.7 points on CLINC150.
3. **Temperature scaling made calibration worse in all three fits.** Fitted correctly, on train-derived validation rows, and it still lost on test each time.
4. **`qwen` failed 8% of its CLINC150 calls outright**, and before the reporter was fixed that failure was being published as seven points of lost accuracy.
5. **`gpt` needed 139 retries across the two tasks** to return 300 answers each, none of which appear in its latency figures.
6. **The reporter had three separate instances of the same defect** and all three would have shipped. Found by an agent reading the raw store against the reporter's output rather than trusting either.

## What this does not answer

- **Whether decision models are order-independent.** The arm was powered to detect an effect the size of Phi's and nothing smaller. Jev's flat rows are consistent with the mechanism and equally consistent with an under-powered test.
- **Whether 4-against-151 isolates option count.** It does not. AG News is topic classification and CLINC150 is intent classification, and this design cannot separate the two.
- **Determinism.** The `--repeat` arm was not run at `--limit 300`; `determinism` in `results/full.json` is empty. The pilot measured it on 2 examples and that number is not carried forward.
- **Whether Laya can do 151 options with a larger head budget.** Deliberately out of scope; that is [`banking77/`](../banking77/README.md).
- **Anything about Gemini or Claude models**, excluded by design; the protocol says why.
- **How any of these models behave outside the cheapest tier of their current generation**, or on hardware other than this one.

## Provenance

| | |
| --- | --- |
| Transport | OpenRouter for every hosted call (`POST /api/v1/chat/completions`; `POST /api/v1/systemone` for Jev); local CPU for Laya. Recorded on every call record, and the report refuses to average across them |
| Data | `fancyzhx/ag_news` default/test (licence **unknown**, per the hub) and `clinc/clinc_oos` plus/test (CC-BY-3.0), read from the hub's parquet, class names read from each file's own `ClassLabel` metadata |
| Run spec | `ag_news` `ad7affdc7ebd`, `clinc150` `40e9e134db8f`, written to `runs/full/` and hash-checked on read |
| Raw wire | `typed_decisions/runs/full/calls.jsonl`, 10,120 records, gitignored: request, response, usage, both clocks, the option order shown, and the distribution where there is one |
| Committed | `typed_decisions/results/full.json` and `results/figures/*.png`, written by `python -m typed_decisions.report --tag full` |
| Seed | 0, for the shared permutation and for the bias orderings |
| Hardware | Windows 11, CPU only; no usable GPU, which is why Laya is grouped apart in every latency table |
| Spend | **$0.7245** over 10,120 calls, against a quote of $0.60 – $1.00 made from the pilot |
| Tests | 487 passing |
| Reporter | Numbers regenerated after `5f70067`; earlier figures for `qwen/clinc150` and `laya/clinc150` are superseded |
| Reproducibility | The report step was run twice from the same store by two people minutes apart. `results/full.json` differed in the `written_at` timestamp and in nothing else, and all five figures came out byte-identical |
| Protocol | [README.md](README.md); prediction committed `f2893da` 2026-09-24 22:37:30, first call record 2026-09-25 08:03 |

**Claims here are measured in this run** unless marked otherwise. The Laya warning and the `head_max_len` error are quoted verbatim from the run's own output and its call records. The licence statuses are read from the Hugging Face dataset cards and are marked as such. The McNemar tests are computed from the same `runs/full/calls.jsonl` the tables are built from, excluding calls that never returned. That `qwen` resolved to `qwen3.8-flash` in this run where the pilot resolved to `qwen3.8-27b` is recorded on the call records; the two runs' `qwen` rows are different models and must not be compared.

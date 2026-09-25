# Results: does Laya's Banking77 failure come from its token budget?

Protocol: [README.md](README.md).

> **The full sweep has not run.** What follows is the pilot: arm A only, 200 test examples, plus the truncation diagnostics, which need no forward pass. Arms B, K, C and E have not been launched, so **the question this experiment asks is not answered yet.**

## The finding

Arm A reproduces the published number, 0.435 accuracy against a published 0.425 and a confidence interval that contains it, and the diagnostics show the mechanism the hypothesis names is real (at the default budget five pairs of labels collapse onto the same string), but nothing here yet says whether raising the budget recovers the accuracy.

## The numbers

### Pilot: arm A on 200 examples

200 test examples drawn from the shared shuffle, covering 68 of the 77 intents.

| | measured | published |
|---|---|---|
| accuracy | **0.435** (95% CI 0.365 to 0.505, 1,000 bootstrap resamples) | 0.425 |
| top-5 accuracy | 0.635 | (not published) |
| macro-F1 | 0.345 | (not published) |
| ECE, 15 bins, raw | 0.336 | (not published) |
| ECE after one fitted temperature (T = 2.17, fitted on 200 validation rows) | 0.120 | (not published) |

The published 0.425 sits inside the confidence interval. **Arm A reproduces.**

### Truncation diagnostics

Read off the sequence `laya.common.build_sequence` itself builds, at the marker positions it returns.

| `head_max_len` | mean text tokens per option | options truncated | options that collapse onto another | identical pairs |
|---|---|---|---|---|
| **256 (arm A)** | 2.74 | 35.1% | 7 | **5** |
| 384 | 3.32 | 0% | 0 | 0 |
| 512 | 3.32 | 0% | 0 | 0 |
| 768 | 3.32 | 0% | 0 | 0 |

The five pairs the model cannot possibly separate at the default budget:

```
" balance not updated"  <-  balance not updated after bank transfer
                            balance not updated after cheque or cash deposit
" lost or stolen"       <-  lost or stolen card
                            lost or stolen phone
" top up by"            <-  top up by bank transfer charge
                            top up by card charge
                            top up by cash or cheque
```

**At 384 nothing truncates at all**, so 384, 512 and 768 build a byte-identical request. The budget sweep has two distinct points, not four. All three are still run when the full sweep goes, because a prediction that three arms will agree is worth checking rather than asserting.

### Latency

**Every latency here is CPU.** The pilot's recorded p50 is 2,093 ms and p95 is 2,843 ms, but the machine was contended for the first 150 calls; over the last 200 calls the p50 is **344 ms** and the p95 is **460 ms**. Both are reported because the first number is what the run actually measured and the second is what the machine does when nothing else is running.

### One complete record

The pilot's wire log holds the exact request, the exact response and the full 77-way distribution for every prediction. The record shown in [README.md](README.md) is one of them: **0.92 probability on the wrong intent, 0.0017 on the right one.** The over-confidence is the model card's own documented behaviour, and it is what the fitted temperature is for.

Raw numbers: `banking77/results/pilot.json`. The wire log is `banking77/runs/pilot/decisions.jsonl`, which is not committed.

## Did the prediction hold?

**Partly, and only the parts the pilot could test.**

- **Held: the three raised-budget arms build a byte-identical request.** `banking77/budget.py` and commit `47eae3b` predicted that at 384 nothing truncates, so 384, 512 and 768 cannot differ. The diagnostics confirm it without a forward pass. The arms are still all run rather than asserted equal.
- **Held: `identical_pairs` is non-zero at the published configuration.** The mechanism the hypothesis names is real: five pairs of labels are indistinguishable to the model at the default budget, and 35.1% of options are truncated.
- **Not tested: whether room alone recovers accuracy.** That is arms B-384, B-512 and B-768, and they have not run. The falsification condition stated in the protocol (the B arms failing to move accuracy while the K arms do) cannot be evaluated yet.
- **Not tested:** the option-count sweep, the coarse-to-fine workaround, and the English checkpoint. Arms K, C and E have not run.

## What surprised us

- **laya's per-option budget has a floor of four tokens, and the floor can overrun the budget it enforces.** `per = max(4, (head_max_len - 16) // n_options)` means the English 192 and the multilingual 256 hand 77 options exactly the same room, which the model card presents as two different budgets. And 77 times 4 is 308, which does not fit in 192 at all, so the instruction text is squeezed to 8 tokens instead. This was found by reading `build_sequence` rather than by measuring, and it was found because one of the 76 tests failed: the test was wrong, not the code.
- **The first pilot had to be thrown away.** `--limit` took the first 200 rows of a split that ships ordered by label, so it measured five intents and called the result accuracy. The seeded shuffle that fixes it is shared by every arm, so no arm is measuring option or example order. The mistake was caught by the label count.

## What this does not answer

- **The question the experiment exists to ask.** Whether raising `head_max_len` recovers the accuracy is arm B, and arm B has not run.
- **Whether accuracy tracks tokens per option or task difficulty.** That is the K arms, unrun. Without them, a gain from a raised budget could not be separated from a gain from an easier task.
- **Whether the documented coarse-to-fine workaround beats raising the budget.** Arm C, unrun.
- **How much of the published number is the checkpoint rather than the budget.** Arm E, unrun. The model card contradicts itself about which checkpoint produced the 0.425, which is why arm E exists.
- **Anything about Jev.** This experiment does not measure it. The published 0.870 was scored on 72 labels against Laya's 77 and is not the same task.
- **Anything at 3,080 examples.** The pilot's interval runs from 0.365 to 0.505, which is wide enough to contain the published number and a good deal else.
- **Anything on a GPU.** Every latency here is CPU, and Convai's published 33 ms is a T4 figure.

## Provenance

| | |
|---|---|
| Model | `convaiinnovations/laya`, multilingual subfolder (responds as `laya-rl-agent`), weights at `C:\projects\jev_demo\arena\models\laya` |
| Configuration | `head_max_len` 256, `max_len` 1024, 77 options, CPU |
| Dataset | BANKING77 from `PolyAI-LDN/task-specific-datasets`, `banking_data/{train,test}.csv` |
| Seed | 20260924, shared shuffle and stratified split; `val_frac` 0.1; 20 warm-up calls |
| Library versions | laya 0.3.7, torch 2.14.0+cpu, transformers 5.17.0, Python 3.12.14 |
| Hardware | Intel64 Family 6 Model 186 Stepping 2 (GenuineIntel), AMD64, Windows-11-10.0.26200-SP0, `cuda_available: false` |
| Run written at | `2026-09-24T11:52:53` (`banking77/runs/pilot/config.json`) |
| Commit | `47eae3b`, which is where the harness and this pilot's results landed. The run itself records no commit |
| Results file | `banking77/results/pilot.json`; diagnostics in `banking77/runs/pilot/diagnostics/A-default.json` |

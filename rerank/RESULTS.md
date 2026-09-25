# Results: can a decision model re-rank retrieval better than BM25?

Protocol: [README.md](README.md).

## The finding

Yes for the hosted decision model and for the cross-encoder, no for the open-weights one: over all 323 NFCorpus test queries Jev moved nDCG@10 by **+0.0347** against the BM25 floor and the cross-encoder by **+0.0204**, while both Laya checkpoints re-ranked **worse than doing nothing**, and all four paired intervals exclude zero.

## The numbers

**All 323 NFCorpus test queries, top 20 candidates each**, official qrels. 6,460 scoring calls per re-ranker, 25,840 in total.

| Ranking | nDCG@10 | Recall@10 | MRR@10 | nDCG@10 vs BM25, paired | p50 | p95 |
|---|---|---|---|---|---|---|
| Jev `score` | **0.3414** | **0.1571** | 0.5624 | **+0.0347** [+0.0218, +0.0476] | 331 ms | 677 ms |
| cross-encoder MiniLM-L-6 | 0.3272 | 0.1499 | **0.5656** | **+0.0204** [+0.0092, +0.0317] | 38 ms | 76 ms |
| BM25 (the floor) | 0.3067 | 0.1523 | 0.5092 | (the floor) | (none) | (none) |
| Laya `score` | 0.2848 | 0.1409 | 0.4791 | **-0.0219** [-0.0368, -0.0070] | 935 ms | 1,251 ms |
| Laya `typed-decisions` `score` | 0.2707 | 0.1389 | 0.4572 | **-0.0360** [-0.0522, -0.0197] | 916 ms | 1,111 ms |

**Every interval excludes zero, in both directions.** That is what this run was for. At 30 queries nothing but Laya's loss was separable from noise; at 323 the four results are all real:

- **Jev re-ranks better than BM25**, +0.0347 nDCG@10, better on 117 queries and worse on 52.
- **The cross-encoder also re-ranks better than BM25**, +0.0204, better on 97 and worse on 67. The pilot could only say it did not hurt; now it helps.
- **Both Laya checkpoints re-rank worse than BM25.** Zero-shot, on this dataset, with these questions, re-ranking with Laya is worse than doing nothing. The base checkpoint is better on 82 queries and worse on 111; the fine-tune is better on 77 and worse on 128.

Jev is ahead of the cross-encoder by 0.0143 nDCG@10, but that pair was not compared directly, and two intervals against a common floor do not settle it.

**Jev's failures were not smoothed over.** 313 of 6,460 calls (4.85%) never succeeded after five retries: 310 `HTTP 503`, 2 `HTTP 429`, 1 `HTTP 504`, against 14 of 1,500 (0.9%) in the pilot. Those 313 passages kept their BM25 position and touch 147 of the 323 queries, up to 7 in a single query. Nothing was invented for them. There were 6,414 retries across the run, and the exact cost from the gateway's own per-call `marketCost` was **$0.1533**, or $0.0237 per 1,000 calls.

Raw numbers: `results/test-top20-q323-*.json`. All 25,840 scoring calls are in `runs/test-top20-q323/scores.jsonl`, request and response included.

### The pilot, kept because it is what the full run was designed to test

30 queries, top 50 candidates each, NFCorpus **test** split with the official qrels. 1,500 scoring calls per re-ranker, seven re-rankers.

| Ranking | nDCG@10 | Recall@10 | MRR@10 | nDCG@10 vs BM25, paired | p50 | p95 |
|---|---|---|---|---|---|---|
| Jev `score` | **0.2998** | **0.1337** | 0.5228 | +0.0247 [-0.0067, +0.0561] | 354 ms | 557 ms |
| cross-encoder MiniLM-L-6 | 0.2930 | 0.1310 | **0.5500** | +0.0180 [-0.0141, +0.0500] | 27 ms | 34 ms |
| Jev `boolean` | 0.2885 | 0.1304 | 0.5207 | +0.0135 [-0.0171, +0.0441] | 343 ms | 552 ms |
| BM25 (the floor) | 0.2751 | 0.1280 | 0.5042 | (the floor) | (none) | (none) |
| Laya `score` | 0.2111 | 0.1036 | 0.3507 | **-0.0640** [-0.1155, -0.0125] | 846 ms | 1,034 ms |
| Laya `noul` | 0.2046 | 0.1094 | 0.3801 | **-0.0705** [-0.1226, -0.0183] | 743 ms | 1,089 ms |
| Laya `typed-decisions` `noul` | 0.1928 | 0.1022 | 0.3668 | **-0.0823** [-0.1481, -0.0165] | 789 ms | 4,062 ms |
| Laya `typed-decisions` `score` | 0.1804 | 0.0994 | 0.3072 | **-0.0946** [-0.1617, -0.0276] | 934 ms | 7,073 ms |

- **Zero-shot Laya was already below the BM25 floor, and the gap was real.** All four Laya intervals sit entirely below zero.
- **Nothing above the floor separated from noise at 30 queries.** Jev's +0.0247 and the cross-encoder's +0.0180 both crossed zero, so the pilot established only that they do not hurt. That is precisely why the full-scale run exists.
- **Neither question formulation ranked better**, for either model. `score` minus `noul`, paired on Laya: **+0.0065** nDCG@10, 95% CI [-0.0384, +0.0514]. The same held for Jev. That is why the full run carries `score` only.
- **The speed argument does not survive contact with a CPU.** Laya is 27 to 31 times *slower* per call than the MiniLM cross-encoder here, 846 ms against 27 ms. Laya's published 39.5 ms is a T4 number, and the cross-encoder is a 6-layer 22M-parameter model against Laya's 421M, so it is a fair fight only on hardware both can use. On CPU there is no economic argument for the local decision model here at all.

The pilot put Laya `score` 0.064 below the floor; at 323 queries and top-20 the loss is 0.022. The direction held, the magnitude did not. Where the two disagree, the full-scale number is the one to believe, and the disagreement is the reason to distrust any single small run, including this one.

Raw numbers: `results/test-top50-q30-*.json`. Every one of the 10,500 scoring calls behind them is in `runs/test-top50-q30/scores.jsonl`.

## Did the prediction hold?

**There is nothing here to check.** No prediction was recorded before this run, and the reconstruction in [README.md](README.md) was written after the answer was known. Comparing a result against a prediction written after it establishes nothing, so this section reports no verdict rather than manufacturing one. The next experiment in this repository gets its prediction committed before its run, which is the whole point of the two-file split.

## What surprised us

**The `typed-decisions` fine-tune ranks below the base checkpoint, and it is not a loading bug.** It was the first thing checked, because a fine-tune losing to its own base contradicts Convai's model card. The two checkpoints are different files (`model.safetensors`, 842,609,210 bytes against 842,609,220, different SHA-256) with different configs (`model_name` `rl-agent` against `laya-typed-decisions`, `max_len` 512 against 1024), `laya.load(..., subfolder=...)` raises rather than falling back if the subfolder is missing, and **all 6,460 (query, passage) pairs scored differently between the two variants**, not one identical value. The fine-tune is genuinely loaded and genuinely worse.

What it does instead is compress the scale: the base checkpoint spreads its answers over 0.082 to 3.5478 (stdev 0.5155), the fine-tune over 1.4319 to 3.3408 (stdev 0.3324). It never uses the bottom third of the scale, so it separates candidates less, and a re-ranker that separates less ranks worse. The `RuntimeWarning: this checkpoint ships invalid temperatures` is a red herring: it names `choice:11+=0.1006 -> 0.5`, a bucket that is identical in **both** configs and belongs to a question type (`choice`) that this benchmark never asks. Both checkpoints emit it.

**Jev answers in coarse steps, and a quarter of its ranking is still BM25's.** The gateway rounds to two decimals, so a 0-to-4 score has at most 401 possible values, and Jev used 389 of them across 6,147 successful calls. **1,748 of those 6,147 scored passages (28%) sit in a tie**, a median of 4 of 20 per query, 10 or more of 20 on 74 queries, and 19 of 20 on the worst one. So Jev's +0.0347 is earned while roughly a quarter of the ranking is still BM25's. That cuts both ways and this run does not separate them: it may mean Jev is decisive exactly where it matters, or it may mean the measured gain comes from fewer decisions than the call count suggests. The cross-encoder, by contrast, emits a raw logit and tied only 136 of 6,460.

## What this does not answer

- **Whether fine-tuning closes a gap this large.** Zero-shot Laya is 0.022 nDCG@10 *below* a bag-of-words baseline, so a fine-tune has to recover that before it wins anything. The one fine-tune available, `typed-decisions`, made it worse rather than better.
- **Why the `typed-decisions` fine-tune ranks below its own base.** It is genuinely the checkpoint being loaded and it genuinely uses a narrower slice of the scale, but nothing here explains why a model fine-tuned for typed decisions discriminates less on this task than the checkpoint it came from. Convai's model card says the opposite should happen.
- **Whether Jev actually beats the cross-encoder.** Both beat BM25 with intervals clear of zero, and Jev is 0.0143 ahead, but that pair was never compared directly. The paired Jev-minus-cross-encoder difference would settle it.
- **What Jev's ties are worth.** 28% of its scored passages sit in a tie and hold BM25's order. Whether the gain comes from decisive judgements on the rest, or whether finer resolution would add more, is untested: the gateway's two-decimal rounding is not something this demo can turn off.
- **Why Laya loses.** The scores vary and look sensible one at a time, but nothing here separates "the checkpoint cannot judge relevance" from "the question is wrong" from "1,000 characters of a medical abstract is not enough to judge on".
- **Whether a better question helps.** Two formulations were tried, not twenty, and no prompt was tuned against the test split.
- **Why 4.85% of Jev's calls failed.** Almost all were `HTTP 503` surviving five retries with exponential backoff, five times the pilot's rate. Whether that is load on the day, the sustained request rate, or something about these payloads is not established here.
- **Whether the 1,000-character passage cut matters.** Nothing here measures what the tail would have added.
- **What this costs on a GPU.** Every timing here is CPU-only.
- **Whether the 91 queries with nothing relevant in the candidate pool are telling us more about NFCorpus than about re-ranking.**

## Provenance

| | |
|---|---|
| Dataset | BEIR NFCorpus test split, 3,633 documents, 323 queries, official qrels, via `BeIR/nfcorpus` and `BeIR/nfcorpus-qrels` |
| Jev | `typesafe-ai/jev`, v4 evaluation-model route over the Vercel AI Gateway |
| Laya | `convaiinnovations/laya`, base English checkpoint (responds as `laya-rl-agent`) and the `typed-decisions` subfolder |
| Cross-encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Commit | `0bd0470`, recorded by the run itself in the results file |
| Finished | full run `2026-09-24T00:16:10+00:00`; pilot files stamped `20260923-093316` through `20260923-140116` |
| Hardware | CPU, `device: "cpu"` in the results file. No usable GPU on this machine |
| Library versions | **not recorded.** This run predates the per-run `config.json` that the `banking77/` and `typed_decisions/` harnesses write, and its results files carry no version block |
| Results files | `results/test-top20-q323-*.json` (full run), `results/test-top50-q30-*.json` (pilot) |
| Wire log | `runs/<tag>/scores.jsonl`, not committed |

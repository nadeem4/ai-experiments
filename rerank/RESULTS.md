# Results: can a decision model re-rank retrieval better than hybrid search?

What the experiment is and how to run it: [README.md](README.md). What the data holds: [DATASET.md](DATASET.md).

## The finding

**Jev, the hosted decision model, makes the ranking better, and better than a cross-encoder does. The open-weights Laya makes it worse than leaving it alone.**

Over all 323 NFCorpus test queries, Jev lifted nDCG@10 from **0.331 to 0.379**, **+0.0478**, 14.5% over the retriever's own order. Measured directly against the MiniLM cross-encoder on the same queries, Jev is ahead by **+0.0319 [+0.0188, +0.0450]**. The cross-encoder helps a little (+0.0159). Both Laya checkpoints come out **negative**, at −0.0241 and −0.0278, and are indistinguishable from each other. All four paired intervals against the floor exclude zero, the cross-encoder's only just: corrected for the ten pairwise nDCG@10 comparisons the report makes, its gain crosses zero (a 99.5% t interval over `results/gpu/per-query.csv`, [−0.0019, +0.0338]; computed from that file, not written by the report). Jev clears the floor on all eight measures now reported and beats the cross-encoder on six; on whether the top result is relevant, and on MRR@10, the two cannot be told apart.

## The numbers

323 queries, 20 hybrid candidates each, four scorers: **25,840 calls**, one of which failed. $0.16, on an NVIDIA T4.

| Method | nDCG@10 | vs floor | 95% interval | better | worse | same |
|---|---:|---:|---|---:|---:|---:|
| `hybrid` *(floor)* | 0.3306 | — | | | | |
| **`jev-score`** | **0.3785** | **+0.0478** | [+0.0334, +0.0622] | **145** | 60 | 118 |
| `cross-encoder` | 0.3465 | +0.0159 | [+0.0035, +0.0283] | 102 | 84 | 137 |
| `laya-score` | 0.3065 | **−0.0241** | [−0.0390, −0.0093] | 90 | **125** | 108 |
| `laya-typed-score` | 0.3028 | **−0.0278** | [−0.0436, −0.0120] | 86 | **136** | 101 |

![Every query's change against the floor](results/gpu/figures/per-query-delta.png)

*Each bar is one query, sorted. The shape is the finding: Jev's mass sits above the line and its negative tail is the shortest of the four, while both Laya panels are the mirror image. The cross-encoder is nearly symmetric, which is what a +0.0159 mean looks like up close.*

**The means are not the whole story, and the panels show why.** No method is uniformly better. Jev, the clear winner, still made 60 queries worse. The cross-encoder's 102 wins and 84 losses very nearly cancel, and its small positive mean survives only because its wins are slightly larger than its losses.

### Head to head

Each row above is against the floor, and two intervals that both clear the floor do not say which of the two methods is better. Every method re-ranked the same candidates for the same queries, so they can be compared with each other directly, query by query:

| | difference | 95% interval | first better | second better | same |
|---|---:|---|---:|---:|---:|
| `jev-score` − `laya-typed-score` | +0.0756 | [+0.0583, +0.0929] | 159 | 56 | 108 |
| `jev-score` − `laya-score` | +0.0720 | [+0.0558, +0.0882] | 161 | 46 | 116 |
| `cross-encoder` − `laya-typed-score` | +0.0437 | [+0.0279, +0.0596] | 146 | 76 | 101 |
| `cross-encoder` − `laya-score` | +0.0401 | [+0.0250, +0.0551] | 146 | 65 | 112 |
| `jev-score` − `cross-encoder` | +0.0319 | [+0.0188, +0.0450] | 124 | 67 | 132 |
| `laya-score` − `laya-typed-score` | +0.0036 | [−0.0051, +0.0124] | 102 | 101 | 120 |

Jev beats the cross-encoder on nDCG@10, and every re-ranker beats both Laya checkpoints. The two Laya checkpoints cannot be told apart: the interval straddles zero and they win almost exactly as often as each other. Every pair is in `results/gpu/summary.json` under `head_to_head`.

### It is not only reordering, it is retrieving better

| Method | Recall@10 | MRR@10 |
|---|---:|---:|
| `hybrid` *(floor)* | 0.1612 | 0.5352 |
| `jev-score` | **0.1815** | **0.5905** |
| `cross-encoder` | 0.1660 | 0.5615 |
| `laya-score` | 0.1530 | 0.4910 |
| `laya-typed-score` | 0.1551 | 0.4850 |

Recall@10 rises for Jev because it pulls relevant passages up from positions 11 to 20 into the top 10. MRR@10 rises because the *first* relevant result arrives sooner, which is what a person actually feels. Both Laya checkpoints push relevant passages **down** within the top 10. Whether they push them out of it, this run cannot say: their Recall@10 differences cross zero (see every measure, below).

### Every measure, with its interval

nDCG@10 is the headline, but a reader may ask a narrower question: was the first result right, how soon does the first relevant passage arrive, how much of the top ten is relevant, how good are only the first three or five places. Each has the same paired comparison nDCG@10 has. These five were added after the results were known (see the README's protocol amendments), and with eight measures and ten pairs a borderline interval or two can clear zero by chance, so read them as secondary.

| Method | Top result relevant (success@1) | MRR@10 | Recall@10 | P@10 | MAP@10 | nDCG@3 | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `hybrid` *(floor)* | 0.4458 | 0.5352 | 0.1612 | 0.2443 | 0.1216 | 0.3876 | 0.3617 | 0.3306 |
| `jev-score` | 0.5046 | 0.5905 | 0.1815 | 0.2731 | 0.1514 | 0.4488 | 0.4167 | 0.3785 |
| `cross-encoder` | 0.4892 | 0.5615 | 0.1660 | 0.2464 | 0.1351 | 0.4136 | 0.3793 | 0.3465 |
| `laya-score` | 0.3839 | 0.4910 | 0.1530 | 0.2362 | 0.1082 | 0.3413 | 0.3214 | 0.3065 |
| `laya-typed-score` | 0.3839 | 0.4850 | 0.1551 | 0.2331 | 0.1042 | 0.3358 | 0.3231 | 0.3028 |

Every method against the floor, paired per query. `~` marks an interval that crosses zero:

| Against the floor | `jev-score` | `cross-encoder` | `laya-score` | `laya-typed-score` |
|---|---|---|---|---|
| Top result relevant (success@1) | +0.0588 [+0.0186, +0.0991] | +0.0433 [+0.0052, +0.0815] | −0.0619 [−0.1116, −0.0123] | −0.0619 [−0.1130, −0.0108] |
| MRR@10 | +0.0554 [+0.0252, +0.0855] | +0.0263 [−0.0011, +0.0538] ~ | −0.0441 [−0.0795, −0.0088] | −0.0502 [−0.0871, −0.0132] |
| Recall@10 | +0.0203 [+0.0109, +0.0298] | +0.0049 [−0.0018, +0.0115] ~ | −0.0081 [−0.0165, +0.0002] ~ | −0.0061 [−0.0163, +0.0041] ~ |
| P@10 | +0.0288 [+0.0196, +0.0380] | +0.0022 [−0.0071, +0.0114] ~ | −0.0080 [−0.0177, +0.0016] ~ | −0.0111 [−0.0212, −0.0010] |
| MAP@10 | +0.0298 [+0.0179, +0.0418] | +0.0135 [+0.0037, +0.0234] | −0.0135 [−0.0242, −0.0027] | −0.0174 [−0.0296, −0.0051] |
| nDCG@3 | +0.0611 [+0.0351, +0.0872] | +0.0260 [+0.0032, +0.0487] | −0.0464 [−0.0759, −0.0169] | −0.0519 [−0.0815, −0.0223] |
| nDCG@5 | +0.0549 [+0.0350, +0.0749] | +0.0176 [+0.0003, +0.0348] | −0.0404 [−0.0629, −0.0179] | −0.0387 [−0.0609, −0.0165] |
| nDCG@10 | +0.0478 [+0.0334, +0.0622] | +0.0159 [+0.0035, +0.0283] | −0.0241 [−0.0390, −0.0093] | −0.0278 [−0.0436, −0.0120] |

Jev against the cross-encoder on each measure:

| `jev-score` − `cross-encoder` | difference | 95% interval | Jev better | cross-encoder better | same |
|---|---:|---|---:|---:|---:|
| Top result relevant (success@1) | +0.0155 | [−0.0270, +0.0580] ~ | 27 | 22 | 274 |
| MRR@10 | +0.0290 | [−0.0012, +0.0592] ~ | 58 | 41 | 224 |
| Recall@10 | +0.0155 | [+0.0071, +0.0239] | 87 | 27 | 209 |
| P@10 | +0.0266 | [+0.0176, +0.0356] | 87 | 27 | 209 |
| MAP@10 | +0.0163 | [+0.0065, +0.0261] | 129 | 59 | 135 |
| nDCG@3 | +0.0352 | [+0.0125, +0.0578] | 75 | 55 | 193 |
| nDCG@5 | +0.0374 | [+0.0197, +0.0551] | 99 | 56 | 168 |
| nDCG@10 | +0.0319 | [+0.0188, +0.0450] | 124 | 67 | 132 |

**Jev clears the floor on all eight.** The cross-encoder does on the measures weighted to the first few places (success@1, nDCG@3, nDCG@5, MAP@10, nDCG@10) and not on MRR@10, Recall@10 or P@10: it reorders the top of the list but rarely brings a relevant passage up from below tenth. Jev does both, which is where its lead over the cross-encoder comes from. On the first result alone the two cannot be told apart: Jev's top result is relevant for 50.5% of queries, the cross-encoder's for 48.9%, an interval across zero. Both Laya checkpoints are worse than the floor on six of the eight and indistinguishable from each other on all eight. Every comparison is in `results/gpu/summary.json` under `by_metric`.

### What each one costs

| Method | p50 | p95 | Whole pass | Cost |
|---|---:|---:|---:|---|
| `cross-encoder` | **6.8 ms** | 10.7 ms | 48 s | free, local |
| `laya-score` | 37.1 ms | 54.0 ms | 257 s | free, local |
| `laya-typed-score` | 37.0 ms | 53.4 ms | 255 s | free, local |
| `jev-score` | 297.8 ms | 392.6 ms | 2,069 s | **$0.1603** |

Jev's per-call figure is a **network round trip**, not the same measurement as a local forward pass, and it is not a claim about how fast the model is. The three local scorers all ran on the same T4 and are comparable with each other.

$0.1603 over 6,460 calls is **$0.0248 per thousand**. The cross-encoder delivers a third of Jev's gain for nothing, 44 times faster.

**Per query, the cost that matters is time, not money.** Re-ranking one query is 20 calls. Made one after another, Jev's take about **6.4 s** (2,069 s of call latency over 323 queries); the cross-encoder's take 0.15 s. Whether Jev's 20 calls can run concurrently, and what that does to the provider's latency, was not measured, and it decides whether Jev fits in front of a person waiting for search results.

### A fifth of Jev's ranking is not Jev's

| Method | Distinct scores | Tied passages | Worst tie |
|---|---:|---:|---:|
| `cross-encoder` | 6,363 of 6,460 | 190 | 2 |
| `laya-score` | 5,246 of 6,460 | 204 | 2 |
| `laya-typed-score` | 4,955 of 6,460 | 204 | 2 |
| **`jev-score`** | **395 of 6,459** | **1,450** | **11** |

Jev's answers land on a two-decimal grid, so a 0-to-4 score has at most 401 places to go, and it used 395 of them. **1,450 of its 6,459 scored passages sit in a tie**, one query having eleven passages on the same value, and a tie keeps the retriever's order. So the +0.0478 is earned while roughly a fifth of the ranking credited to it is still the first stage's. That cuts both ways and this run does not separate them: Jev may be decisive exactly where it matters, or the gain may come from fewer real decisions than 6,460 calls suggests.

![How many different answers each scorer gave](results/gpu/figures/score-distribution.png)

### The sample that could not move

Of the 323 queries, **71 have nothing relevant among their 20 candidates**, so every method scores zero on them whatever order it picks. The means above are over all 323, and the per-query table marks which could move. Across all queries the candidates hold **3.64 relevant passages on average**, and at most 20.

Those 71 dilute every mean equally. Over the **252 queries that could move**, the same comparisons read:

| Method | vs floor | 95% interval | better | worse | same |
|---|---:|---|---:|---:|---:|
| **`jev-score`** | **+0.0613** | [+0.0432, +0.0794] | **145** | 60 | 47 |
| `cross-encoder` | +0.0204 | [+0.0046, +0.0362] | 102 | 84 | 66 |
| `laya-score` | −0.0310 | [−0.0500, −0.0119] | 90 | 125 | 37 |
| `laya-typed-score` | −0.0356 | [−0.0558, −0.0155] | 86 | 136 | 30 |

Jev against the cross-encoder on the same 252 is +0.0409 [+0.0243, +0.0575]. Nothing changes order or sign; every gap is about a quarter wider, which is what removing 71 zeros from the denominator should do.

## Did the prediction hold?

**There is nothing here to check.** No prediction was recorded before this run. Comparing a result against an expectation recalled afterwards establishes nothing, so this reports no verdict rather than manufacturing one.

## What surprised us

1. **The fine-tune is no better than the checkpoint it was tuned from.** `laya-typed-score` is −0.0278 against `laya-score`'s −0.0241, which looks worse, but the direct comparison is +0.0036 [−0.0051, +0.0124] in the base's favour: an interval across zero, with each winning on 101 or 102 queries. An earlier version of this file said the fine-tune was worse; this run cannot show that. Whatever the `typed-decisions` tune optimised, it did not help here.

2. **Both open-weights checkpoints are worse than doing nothing at all.** Not merely behind Jev, behind the retriever. A pipeline would be better off deleting them than running them, which is a stronger statement than "underperforms" and is the one the interval supports.

3. **The query the protocol uses as its worked example is one Jev gets wrong.** On `PLAIN-102`, "Stopping Heart Disease in Childhood", the retriever put the one relevant abstract second. Jev's order lowers nDCG@10 by 0.1022 and the cross-encoder's raises it by 0.1195. It is a fair picture of the 60 queries Jev makes worse: the winner on average is not the winner on every question.

4. **Jev improves recall@10 as well as ordering**, 0.1612 to 0.1815. It is not only rearranging the top ten, it is pulling relevant passages into it from below.

5. **At the very top, Jev's lead disappears.** It beats the cross-encoder on nDCG@10, Recall@10, P@10, MAP@10, nDCG@3 and nDCG@5, but not on whether the first result is relevant (+0.0155 [−0.0270, +0.0580]) or on MRR@10. Its advantage is in the rest of the top ten, not the first place.

6. **Jev's resolution is its own ceiling.** 395 distinct values is not a rounding detail when 1,450 passages tie because of it.

## What this does not answer

- **Whether Jev is fast enough to sit in front of a person.** Twenty calls a query, about 6.4 s if made in turn. Concurrency was not tested.
- **Whether Jev beats a larger cross-encoder.** The one here is the smallest MiniLM, 22M parameters. Jev beats it; a bigger cross-encoder is the comparison a production team would actually make.
- **What Jev's ties are worth.** A fifth of its scored passages hold the retriever's order. Whether finer resolution would add more is untested, and the two-decimal rounding is not something a client can switch off.
- **Whether a fine-tune would rescue Laya.** This is zero-shot by design. The number here is the baseline that would make a later fine-tune interpretable, not a verdict on the architecture.
- **Anything beyond 20 candidates.** Recall@10 is bounded by what the first stage retrieved, and 71 queries had nothing relevant to find.
- **Whether this transfers off NFCorpus.** One corpus, one domain: medical abstracts against health questions.

## Where the output lives

| | |
|---|---|
| `results/gpu/summary.json` | every metric above, the head-to-head comparisons, every measure's comparisons (`by_metric`), the 252-query view, and the run's provenance |
| `results/gpu/methods.csv` | the headline table, one row per method |
| `results/gpu/per-query.csv` | 323 rows: the floor's nDCG@10, each method's delta, whether the query could move |
| `results/gpu/scores.csv` | every distinct score value and how many calls landed on it |
| `results/gpu/latency.csv` | one row per successful call |
| `results/gpu/figures/*.png` | generated by the report step from the files above |
| `runs/gpu/scores.jsonl` | the raw wire log, 25,840 records, gitignored |
| `runs/gpu/candidates.json` | the pinned candidate set every method re-ranked |
| `runs/gpu/config.json` | the retrieval settings, the device and the commit |

## Provenance

| | |
|---|---|
| Data | BEIR NFCorpus, official test split: 3,633 documents, 323 queries, official qrels |
| First stage | hybrid: BM25 and `all-MiniLM-L6-v2` each to depth 100, fused by reciprocal rank at `k = 60`, top 20 kept. 39.7 s |
| Models | Jev over OpenRouter's `systemone` route; Laya's base English checkpoint and its `typed-decisions` fine-tune locally; `cross-encoder/ms-marco-MiniLM-L-6-v2` locally |
| Hardware | NVIDIA T4. `laya` runs float16 on CUDA and would run float32 on a CPU |
| Interval | 95% t interval over per-query differences, paired |
| Calls | 25,840, of which 1 failed. That passage kept the retriever's position; nothing was invented for it |
| Spend | $0.1603, from the provider's own per-call figure rather than tokens times a rate |
| Commit | `a41a908`, recorded by the run rather than by the report |
| Finished | the run, 2026-09-26T18:29:46Z. The report was regenerated on 2026-10-06 to add the head-to-head and 252-query comparisons, so that is the timestamp `summary.json` now carries |
| Reproduces | re-running the report over the same wire log reproduces every method row exactly, differing only in the timestamp. Checked again on 2026-10-06, twice: after the head-to-head comparisons and after the further measures were added, the CSV tables came back byte-identical |

**Claims here are measured in this run.** The paired differences are computed over the queries both a method and the floor scored, and the failed call is reported rather than dropped.

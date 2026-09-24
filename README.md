# AI Experiments

Measured experiments on AI models, each one carrying the raw wire it was measured from. **Published at [lab.codewithnk.com](https://lab.codewithnk.com).** A sibling repo runs the [Decision Arena](https://arena.codewithnk.com), where the same models play games.

The first experiment is the whole of this repo so far.

---

## Re-ranking retrieval with a decision model

A RAG re-ranker driven by a decision model, evaluated on a public retrieval benchmark.

### What a decision model is

It does not write text. You hand it the situation in words and the options you will accept, and it hands back a number for every option — a probability, or a position on a scale you defined. There is nothing to parse, and it cannot answer with something that is not on the list.

Two of them are measured here. **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** is TypeSafe AI's hosted System One model, closed weights, reached over HTTP. **[Laya](https://github.com/NandhaKishorM/laya)** is Convai's open-weights alternative, 421M parameters, downloaded and run on your own machine. Both answer the same typed questions, so the same request can be sent to either.

### What this experiment asked

Does putting a decision model in front of a retrieval ranking make the ranking better, and what does it cost?

BM25 retrieves a fixed set of candidate passages per query. Each candidate is then handed to a model as **one typed question** — query and passage in, one number out — and the candidates are re-sorted by that number. The question is whether the ranking gets better, and how much the re-ranking costs in wall clock and in money.

**Nothing is trained here.** This is deliberately zero-shot: the numbers below are the baseline that makes a later fine-tune interpretable. Laya's own model card is blunt that its base checkpoints are near chance on typed decisions zero-shot, so a low number for Laya is expected information, not a bug.

Four re-rankers are measured against the BM25 floor: **Jev** over HTTP, a **MS MARCO cross-encoder** locally, and **Laya** on two checkpoints — the base English one and the `typed-decisions` fine-tune. Jev and the cross-encoder both beat the floor; both Laya checkpoints lose to it.

## How it works

```mermaid
%%{init: {'theme': 'neutral'}}%%
flowchart LR
    Q[query] --> B[BM25 over 3,633 documents]
    B --> C[top-k candidates]
    C --> R[reranker.score: query + passage]
    R --> S[one number per passage]
    S --> O[re-sorted ranking]
    O --> M[pytrec_eval: nDCG@10, Recall@10, MRR@10]
    R --> W[scores.jsonl: the exact request and response]
```

Every method re-ranks the **same** candidate list, so the comparison is fair and BM25 itself is the floor.

- **`rerank/data.py`** loads BEIR NFCorpus — 3,633 documents, 323 test queries — from the Hugging Face mirror of BEIR's own files (`BeIR/nfcorpus` for the corpus and queries, `BeIR/nfcorpus-qrels` for the official `test.tsv`), so the numbers are comparable to published work. Not the `beir` package: it pulls sentence-transformers, elasticsearch and a torch stack in just to unzip a dataset, and it fetches a zip from a university host that is regularly down. `ir_datasets` would have worked too; `hf_hub_download` gets the identical official files with one small dependency, a cache and a resumable download.
- **`rerank/bm25.py`** builds the candidates. The order it returns is both the BM25 ranking and the tie-break every other method falls back to.
- **`rerank/rerankers/`** has one module per model, all with the same interface, `score(query, passage) -> {score, request, response, latency_ms}`. `make_reranker(name)` in `__init__.py` is the only place a reranker is chosen by name, so adding Jev is a new module and one line — the task does not change.
- **`rerank/rank.py`** turns scores into a ranking. Ties keep the candidate order, so "no opinion" means "no change" rather than "shuffle".
- **`rerank/metrics.py`** wraps `pytrec_eval` (the trec_eval C code). trec_eval has no MRR@10, so the run is cut to the top 10 before `recip_rank` is asked for. `mean_ci` is a plain t interval written out rather than pulled from scipy, which would be a large dependency for twenty lines.
- **`rerank/store.py`** is the append-and-flush JSONL log, which doubles as the resume point.
- **`rerank/evaluate.py`** is the task and the CLI.

## Every decision is on the wire

A re-ranker that emits only a score is not good enough here. Every (query, passage) pair scored appends one line to `runs/<tag>/scores.jsonl` holding the exact request sent and the exact response received:

```json
{
  "method": "laya-noul",
  "query_id": "PLAIN-102",
  "doc_id": "MED-3253",
  "score": 0.1733,
  "latency_ms": 621.51,
  "request": {
    "state": {
      "query": "Stopping Heart Disease in Childhood",
      "passage": "Pathobiological determinants of atherosclerosis in youth risk scores..."
    },
    "questions": {
      "relevance": {
        "type": "noul",
        "instructions": "A search engine returned this passage for this query. Does this passage answer the query?"
      }
    }
  },
  "response": {
    "model": "laya-rl-agent",
    "answers": {
      "relevance": {"type": "noul", "noul": 0.1733, "confidence": 0.8267, "action": {"act_probability": 1.0}}
    },
    "usage": {"input_tokens": 233, "output_tokens": 0}
  }
}
```

That file is also why the run is resumable. Rerun the same command after an interruption and only the unscored pairs are sent; the candidate set is pinned in `runs/<tag>/candidates.json` so a resumed run re-ranks exactly the same passages.

## The two question formulations

Nobody has published which typed question ranks better, and the second pass costs one more sweep, so both are run on identical data.

| Name | Type | What it takes | What it returns |
|---|---|---|---|
| `laya-score` | `score` | an ordered list of five rungs, "irrelevant" to "directly answers the query" | the expected position on that scale, 0 to 4 |
| `laya-noul` | `noul` | no criteria at all | one probability that the passage answers the query |

Both are in `rerank/rerankers/laya.py`. `laya-typed-score` and `laya-typed-noul` ask the identical questions of the `typed-decisions` fine-tune instead of the base English checkpoint — the checkpoint is the only thing that differs. All four read one weights folder: `models/laya`, downloaded on first use, or wherever `LAYA_PATH` points.

`rerank/rerankers/jev.py` asks Jev the same two questions over HTTP. Only the dialect is translated: Jev's v4 schema calls the criteria-free question `boolean` and answers it with `probability` rather than `noul`, so `jev-noul` sends `"type": "boolean"` and reads `probability`. The instructions and the state are word for word what Laya is handed, so the two models are asked the identical thing.

The full-scale run skipped the `noul`/`boolean` variants. The pilot showed they were not better than `score` for either model, and repeating them at 323 queries would have doubled the cost of a question already answered.

## Full-scale results

**All 323 NFCorpus test queries, top 20 candidates each**, official qrels. 6,460 scoring calls per re-ranker, 25,840 in total. Laya and the cross-encoder ran locally on CPU — there is no usable GPU on this machine — so their latencies are CPU latencies and are not comparable to Laya's published 39.5 ms on a T4. Jev ran over the network, so its latency is a round trip and not a comparable quantity at all.

| Ranking | nDCG@10 | Recall@10 | MRR@10 | nDCG@10 vs BM25, paired | p50 | p95 |
|---|---|---|---|---|---|---|
| Jev `score` | **0.3414** | **0.1571** | 0.5624 | **+0.0347** [+0.0218, +0.0476] | 331 ms | 677 ms |
| cross-encoder MiniLM-L-6 | 0.3272 | 0.1499 | **0.5656** | **+0.0204** [+0.0092, +0.0317] | 38 ms | 76 ms |
| BM25 (the floor) | 0.3067 | 0.1523 | 0.5092 | — | — | — |
| Laya `score` | 0.2848 | 0.1409 | 0.4791 | **−0.0219** [−0.0368, −0.0070] | 935 ms | 1,251 ms |
| Laya `typed-decisions` `score` | 0.2707 | 0.1389 | 0.4572 | **−0.0360** [−0.0522, −0.0197] | 916 ms | 1,111 ms |

**Every interval now excludes zero, in both directions.** That is what this run was for. At 30 queries nothing but Laya's loss was separable from noise; at 323 the four results are all real:

- **Jev re-ranks better than BM25**, +0.0347 nDCG@10, better on 117 queries and worse on 52.
- **The cross-encoder also re-ranks better than BM25**, +0.0204, better on 97 and worse on 67. The pilot could only say it did not hurt; now it helps.
- **Both Laya checkpoints re-rank worse than BM25.** Zero-shot, on this dataset, with these questions, re-ranking with Laya is worse than doing nothing.

Jev is ahead of the cross-encoder by 0.0143 nDCG@10, but that pair was not compared directly here — each interval is against the BM25 floor, and two intervals that both exclude zero do not establish that one method beats the other.

**The `typed-decisions` fine-tune ranks below the base checkpoint, and that is not a loading bug.** It was the first thing checked, because a fine-tune losing to its own base contradicts Convai's model card. The two checkpoints are different files (`model.safetensors`, 842,609,210 bytes vs 842,609,220, different SHA-256) with different configs (`model_name` `rl-agent` vs `laya-typed-decisions`, `max_len` 512 vs 1024), `laya.load(..., subfolder=...)` raises rather than falling back if the subfolder is missing, and **all 6,460 (query, passage) pairs scored differently between the two variants** — not one identical value. The fine-tune is genuinely loaded and genuinely worse.

What it does instead is compress the scale: the base checkpoint spreads its answers over 0.082–3.548 (stdev 0.516), the fine-tune over 1.432–3.341 (stdev 0.332). It never uses the bottom third of the scale, so it separates candidates less, and a re-ranker that separates less ranks worse. The `RuntimeWarning: this checkpoint ships invalid temperatures` is a red herring: it names `choice:11+=0.1006 -> 0.5`, a bucket that is identical in **both** configs and belongs to a question type (`choice`) that this benchmark never asks. Both checkpoints emit it.

**Jev answers in coarse steps.** The gateway rounds to two decimals (`rounding: {"scoreDecimals": 2}` comes back on every call), so a 0-to-4 score has at most 401 possible values and Jev used 389 of them across 6,147 successful calls. The consequence is ties, and a tie is not an opinion — `rank_by_score` leaves tied passages in the order BM25 gave them. **1,748 of the 6,147 scored passages (28%) sit in a tie**, a median of 4 of 20 per query, 10 or more of 20 on 74 queries, and 19 of 20 on the worst one. So Jev's +0.0347 is earned while roughly a quarter of the ranking is still BM25's. That cuts both ways and this run does not separate them: it may mean Jev is decisive exactly where it matters, or it may mean the measured gain comes from fewer decisions than the call count suggests. The cross-encoder, by contrast, emits a raw logit and tied only 136 of 6,460.

**Jev's failures are real and were not smoothed over.** 313 of 6,460 calls (4.85%) never succeeded after five retries — 310 `HTTP 503`, 2 `HTTP 429`, 1 `HTTP 504` — against 14 of 1,500 (0.9%) in the pilot. Those 313 passages kept their BM25 position and touch 147 of the 323 queries, up to 7 in a single query. Nothing was invented for them. There were 6,414 retries across the run, and the exact cost from the gateway's own per-call `marketCost` was **$0.1533**, or $0.0237 per 1,000 calls.

Two caveats on the timings. Jev's 3,118 s of scoring is the sum of the attempts that succeeded; real elapsed time was about 3 h 40 m, the difference being retry backoff. And both Laya passes ran while the Jev pass was in flight — Jev is network-bound and Laya is CPU-bound, so they do not contend, but the Laya latencies are slightly pessimistic for it and not strictly comparable to the pilot's. They remain comparable to *each other*, which is what the base-vs-fine-tune question needs.

Raw numbers: `results/test-top20-q323-*.json`. All 25,840 scoring calls are in `runs/test-top20-q323/scores.jsonl`, request and response included.

## Pilot results

30 queries, top 50 candidates each, NFCorpus **test** split with the official qrels. 1,500 scoring calls per re-ranker, seven re-rankers. **Laya and the cross-encoder both ran on CPU** — there is no usable GPU on this machine — so their latencies are CPU latencies and are not comparable to Laya's published 39.5 ms on a T4. Jev ran over the network throughout.

These numbers are kept because they are what the full-scale run was designed to test, not because they agree with it. **They are not directly comparable**: the pilot used top-50 candidates and the full run top-20, and a shallower candidate pool is a cleaner one, which is why BM25's own floor moves from 0.2751 to 0.3067. The pilot also ran the `noul`/`boolean` variants, which the full run skipped — they were not better than `score` for either model, and at full scale they would have doubled the cost of a question already answered.

Where the two disagree, the full-scale number is the one to believe. The pilot put Laya `score` 0.064 below the floor; at 323 queries and top-20 the loss is 0.022. The direction held, the magnitude did not.

| Ranking | nDCG@10 | Recall@10 | MRR@10 | nDCG@10 vs BM25, paired | p50 | p95 |
|---|---|---|---|---|---|---|
| Jev `score` | **0.2998** | **0.1337** | 0.5228 | +0.0247 [−0.0067, +0.0561] | 354 ms | 557 ms |
| cross-encoder MiniLM-L-6 | 0.2930 | 0.1310 | **0.5500** | +0.0180 [−0.0141, +0.0500] | 27 ms | 34 ms |
| Jev `boolean` | 0.2885 | 0.1304 | 0.5207 | +0.0135 [−0.0171, +0.0441] | 343 ms | 552 ms |
| BM25 (the floor) | 0.2751 | 0.1280 | 0.5042 | — | — | — |
| Laya `score` | 0.2111 | 0.1036 | 0.3507 | **−0.0640** [−0.1155, −0.0125] | 846 ms | 1,034 ms |
| Laya `noul` | 0.2046 | 0.1094 | 0.3801 | **−0.0705** [−0.1226, −0.0183] | 743 ms | 1,089 ms |
| Laya `typed-decisions` `noul` | 0.1928 | 0.1022 | 0.3668 | **−0.0823** [−0.1481, −0.0165] | 789 ms | 4,062 ms |
| Laya `typed-decisions` `score` | 0.1804 | 0.0994 | 0.3072 | **−0.0946** [−0.1617, −0.0276] | 934 ms | 7,073 ms |

**Zero-shot Laya is below the BM25 floor, and the gap is real.** Every method re-ranks the same candidates for the same queries, so the comparison is paired per query, and all four Laya intervals sit entirely below zero. Re-ranking with these checkpoints, on this dataset, with these questions, makes retrieval worse than doing nothing.

**Nothing above the floor separated from noise at 30 queries.** Jev's +0.0247 and the cross-encoder's +0.0180 both crossed zero, so the pilot established only that they do not hurt. That is precisely why the full-scale run exists, and at 323 queries both gains became real.

**Neither question formulation ranked better**, for either model. `score` minus `noul`, paired on Laya: **+0.0065** nDCG@10, 95% CI [−0.0384, +0.0514]. The same held for Jev. That is why the full run carries `score` only.

**The speed argument does not survive contact with a CPU.** Laya is 27 to 31 times *slower* per call than the MiniLM cross-encoder here (846 ms against 27 ms). Laya's published 39.5 ms is a T4 number; this machine has no usable GPU, and the cross-encoder is a 6-layer 22M-parameter model against Laya's 421M, so it is a fair fight only on hardware both can use. What this establishes is that on CPU there is no economic argument for the local decision model here at all.

Raw numbers: `results/test-top50-q30-*.json`. Every one of the 10,500 scoring calls behind them is in `runs/test-top50-q30/scores.jsonl`, request and response included.

## Run it

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

### Laya's weights

Laya needs no key. About 2.3 GB of weights download on first use into `models/laya`, which is not committed. If you already have them, point at that folder instead and nothing is downloaded:

```
export LAYA_PATH=/path/to/laya        # or pass --laya-path
```

If the weights are missing **and** the download fails, the run stops and says both of those things rather than half-loading a model.

### Jev's key

Jev is a hosted API and needs a key. Copy `.env.example` to `.env` in this repo's root:

```
AI_GATEWAY_API_KEY=your_key       # Vercel AI Gateway (typesafe-ai/jev)
TYPESAFE_API_KEY=your_key         # optional: TypeSafe's API directly, preferred when set
```

Either variable also works straight from the environment; `TYPESAFE_API_KEY` wins when both are set, as it pins the model version. The Vercel account needs a credit card on file before AI Gateway will serve requests. Nothing but `jev-*` needs a key: BM25, the cross-encoder and both Laya checkpoints run entirely locally.

### The run

```
uv sync
uv run python -m rerank.evaluate --limit 30 --top-k 50
```

The full-scale run above is:

```
uv run python -m rerank.evaluate --limit 0 --top-k 20 \
    --methods bm25 laya-score laya-typed-score jev-score cross-encoder
```

`--limit 0` runs all 323 test queries. `--top-k` sets candidates per query. `--methods` picks a subset of `bm25`, `laya-score`, `laya-noul`, `laya-typed-score`, `laya-typed-noul`, `jev-score`, `jev-noul`, `cross-encoder`. The dataset lands in the Hugging Face cache, or in `--cache-dir`.

Results go to `results/<tag>-<timestamp>.json`; the raw wire log and the candidate set go to `runs/<tag>/` and are not committed.

A method whose pairs are all already in the store is summarized from those records without loading the model, so re-running the command after an interruption resumes rather than restarts, and re-running it when everything is done just re-reports.

```
uv run pytest
```

## The site

`site/` publishes this experiment at [lab.codewithnk.com](https://lab.codewithnk.com), written as an experiment rather than a results dump: the question, why it is worth asking, **the prediction as it was made before the run**, the mechanism, the findings, the boundaries on them, and the conclusions.

Four hand-built SVG charts carry the argument, with no charting library:

- **the strip plot** — all 323 queries as one dot each, four rows, positioned by their nDCG@10 change against BM25, with a vertical line at zero. It shows what the four means hide: the biggest single group of queries is the one where nothing moved. It is also the navigation, because clicking a dot opens that query below.
- **the slope chart** — BM25's 20 positions on the left against the chosen re-ranker's on the right, judged-relevant passages emphasised from the qrels. Switching method animates the re-order, and honours `prefers-reduced-motion`.
- **the quantisation rug** — every distinct value Jev and the cross-encoder returned, each on its own range plus a magnified sixteenth of it, which is where Jev's two-decimal grid becomes visible.
- **the interval plot** — the four paired differences and their 95% intervals against one zero line, so "excludes zero" can be checked rather than believed.

The table, the charts and the CSV are the whole run. The browsable examples are **a curated subset of 24 queries** — the full wire is 25,840 records of roughly 3 KB and cannot be shipped to a browser — chosen by `rerank/examples.py` to span the outcomes, including the queries where Jev's calls failed. That selection is the only judgement call in the export, so it is a pure function with tests. The page says the same thing.

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

## What this does not answer yet

- **Whether fine-tuning closes a gap this large.** Zero-shot Laya is 0.022 nDCG@10 *below* a bag-of-words baseline, so a fine-tune has to recover that before it wins anything. The one fine-tune available, `typed-decisions`, made it worse rather than better.
- **Why the `typed-decisions` fine-tune ranks below its own base.** It is genuinely the checkpoint being loaded and it genuinely uses a narrower slice of the scale, but nothing here explains why a model fine-tuned for typed decisions discriminates less on this task than the checkpoint it came from. Convai's model card says the opposite should happen.
- **Whether Jev actually beats the cross-encoder.** Both beat BM25 with intervals clear of zero, and Jev is 0.0143 ahead, but that pair was never compared directly. Two intervals against a common floor do not settle it; the paired Jev-minus-cross-encoder difference would.
- **What Jev's ties are worth.** 28% of its scored passages sit in a tie and hold BM25's order. Whether the gain comes from decisive judgements on the rest, or whether finer resolution would add more, is untested — the gateway's two-decimal rounding is not something this demo can turn off.
- **Why Laya loses.** The scores vary and look sensible one at a time, but nothing here separates "the checkpoint cannot judge relevance" from "the question is wrong" from "1,000 characters of a medical abstract is not enough to judge on".
- **Whether a better question helps.** Two formulations were tried, not twenty, and no prompt was tuned against the test split — doing so would make the number meaningless.
- **Why 4.85% of Jev's calls failed.** Almost all were `HTTP 503` surviving five retries with exponential backoff, five times the pilot's rate. Whether that is load on the day, the sustained request rate, or something about these payloads is not established here.
- **Whether the 1,000-character passage cut matters.** NFCorpus abstracts are longer than the English checkpoint's ~320-token state budget, so they are cut where it is visible in the recorded request. Nothing here measures what the tail would have added.
- **What this costs on a GPU.** Every timing here is CPU-only.

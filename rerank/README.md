# Can a decision model re-rank retrieval better than BM25?

**Being measured again.** The completed run was measured on this machine's CPU
over the Vercel AI Gateway, whose key no longer exists, so it cannot be
reproduced. Its results files have been removed and it is being re-measured on a
Kaggle GPU. [RESULTS.md](RESULTS.md) still describes the old run and says so at
the top.

This file says what the experiment is and how to run it. Nothing here is a result.

## The question

Does putting a decision model in front of a retrieval ranking make the ranking better, and what does it cost?

## Motivation

A decision model does not write text. You hand it the situation in words and the options you will accept, and it hands back a number for every option: a probability, or a position on a scale you defined. There is nothing to parse, and it cannot answer with something that is not on the list.

Two of them are measured here. **[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)** is TypeSafe AI's hosted System One model, closed weights, reached over HTTP. **[Laya](https://github.com/NandhaKishorM/laya)** is Convai's open-weights alternative, 421M parameters, downloaded and run on your own machine. Both answer the same typed questions, so the same request can be sent to either.

Re-ranking is a real stage in a real pipeline, and today it is served either by a cross-encoder or by a language model call per passage. If a decision model can judge relevance in tens of milliseconds it is cheaper to operate than a cross-encoder and far cheaper than a language model call per candidate, which is the case worth testing.

**Nothing is trained here.** This is deliberately zero-shot: the numbers are the baseline that makes a later fine-tune interpretable. Laya's own model card is blunt that its base checkpoints are near chance on typed decisions zero-shot, so a low number for Laya is expected information, not a bug.

What changes depending on the answer: if a decision model beats BM25, re-ranking is a place to put one, and the cost per thousand calls decides whether it beats the cross-encoder in production. If it does not, the open-weights case needs a fine-tune before it is worth anything here at all.

## What we expect

**Nothing was written down before this run, so this experiment has no prediction to test.**

That is the honest state of it. The expectation at the time was that Laya would re-rank better than Jev, because Convai's published benchmarks put it ahead on text relevance and this is a text relevance task — but that was never recorded anywhere before the answer was known, and an expectation recalled afterwards is not evidence of anything. [RESULTS.md](RESULTS.md) reports no verdict for that reason rather than manufacturing one.

The experiments written after this one commit their predictions before the run they are judged against, and say in their own results files what `git log` shows about that. This section is what that habit exists to prevent.

## Data

[BEIR](https://github.com/beir-cellar/beir) **[NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/)**, the official **test** split: **3,633 documents and 323 test queries**, with the official qrels.

| | source |
|---|---|
| corpus and queries | [`BeIR/nfcorpus`](https://huggingface.co/datasets/BeIR/nfcorpus) (parquet) |
| relevance judgements | [`BeIR/nfcorpus-qrels`](https://huggingface.co/datasets/BeIR/nfcorpus-qrels), official `test.tsv` |

`rerank/data.py` fetches these with `hf_hub_download`. They are the same bytes BEIR publishes, so the numbers are comparable to published work. Not the `beir` package: it pulls sentence-transformers, elasticsearch and a torch stack in just to unzip a dataset, and fetches a zip from a university host that is regularly down. The files land in the Hugging Face cache, or in `--cache-dir`.

NFCorpus qrels grade 0, 1 or 2; anything above 0 counts as relevant. Passages are cut to the first 1,000 characters before scoring, the same cut for every method.

**Licence:** not recorded here. The files carry whatever the BEIR repositories above carry.

## Method

BM25 retrieves a fixed set of candidate passages per query. Each candidate is handed to a model as **one typed question** — query and passage in, one number out — and the candidates are re-sorted by that number.

```mermaid
%%{init: {'theme':'base','themeVariables':{'primaryColor':'#ffffff','primaryTextColor':'#000000','primaryBorderColor':'#000000','lineColor':'#000000','secondaryColor':'#ffffff','tertiaryColor':'#ffffff','background':'#ffffff','mainBkg':'#ffffff','textColor':'#000000'}}}%%
flowchart LR
    Q[query] --> B[BM25 over 3,633 documents]
    B --> C[top-k candidates]
    C --> R[reranker.score: query + passage]
    R --> S[one number per passage]
    S --> O[re-sorted ranking]
    O --> M[pytrec_eval: nDCG@10, Recall@10, MRR@10]
    R --> W[scores.jsonl: the exact request and response]
```

**What varies: the scorer, and nothing else.**

| Method | What it is |
|---|---|
| `bm25` | the floor. The candidate order BM25 itself returned |
| `jev-score` | Jev over HTTP, the `score` question |
| `cross-encoder` | `cross-encoder/ms-marco-MiniLM-L-6-v2`, locally |
| `laya-score` | Laya's base English checkpoint, the `score` question |
| `laya-typed-score` | the `typed-decisions` fine-tune, the identical question |

**What is held fixed.** Every method re-ranks the **same** candidate list for the same queries, so the comparison is paired per query and BM25 is the floor rather than a rival. The same instructions and the same state go to Jev and to Laya, word for word. The same 1,000-character cut applies to Laya and to the cross-encoder. Ties keep the candidate order, so "no opinion" means "no change" rather than "shuffle". No prompt is tuned against the test split.

**Seeds:** none. BM25, the cross-encoder and both Laya checkpoints are deterministic here. Jev is a hosted service and is not under this repository's control.

### The two question formulations

Nobody has published which typed question ranks better, so both were run on identical data in the pilot.

| Name | Type | What it takes | What it returns |
|---|---|---|---|
| `laya-score` | `score` | five rungs, "irrelevant" to "directly answers the query" | the expected position on that scale, 0 to 4 |
| `laya-noul` | `noul` | no criteria at all | one probability that the passage answers the query |

Both live in `rerank/rerankers/laya.py`. `laya-typed-score` and `laya-typed-noul` ask the identical questions of the `typed-decisions` fine-tune instead of the base checkpoint; the checkpoint is the only thing that differs. All four read one weights folder, `models/laya`, downloaded on first use or wherever `LAYA_PATH` points.

`rerank/rerankers/jev.py` asks Jev the same two questions. Only the dialect is translated: Jev calls the criteria-free question `boolean` and answers it with `probability` rather than `noul`, so `jev-noul` sends `"type": "boolean"`. The instructions and the state are word for word what Laya is handed.

### The code

| File | What it does |
|---|---|
| `data.py` | loads the corpus, the queries and the qrels |
| `bm25.py` | builds the candidates. Its order is both the BM25 ranking and every other method's tie-break |
| `rerankers/` | one module per model, all `score(query, passage) -> {score, request, response, latency_ms}`. `make_reranker(name)` is the only place a model is chosen by name |
| `rank.py` | turns scores into a ranking |
| `metrics.py` | wraps `pytrec_eval` (the trec_eval C code). It has no MRR@10, so the run is cut to the top 10 before `recip_rank` is asked for |
| `store.py` | the append-and-flush JSONL log, which doubles as the resume point |
| `evaluate.py` | the task and the CLI |

## Where the output goes

| Path | What | Committed |
|---|---|---|
| `rerank/results/<tag>-<timestamp>.json` | the summary metrics for every method: nDCG@10, Recall@10, MRR@10, the paired differences and their intervals, latency, cost, call accounting | **yes** |
| `rerank/runs/<tag>/scores.jsonl` | the raw wire log. One line per (query, passage) scored, holding the exact request sent and the exact response received | no, gitignored |
| `rerank/runs/<tag>/candidates.json` | the pinned candidate set, so a resumed run re-ranks exactly the same passages | no, gitignored |

The wire log is why the run is resumable: re-run the same command after an interruption and only the unscored pairs are sent. A method whose pairs are all already in the store is summarised from those records without loading the model, so re-running when everything is done just re-reports.

One line of `scores.jsonl`:

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

A re-ranker that emitted only a score would not be enough: without the exchange there is no way to check afterwards what a model was actually asked.

## Metrics

- **nDCG@10**, the primary measure. The standard BEIR headline number, so the floor and the methods can be read against published work.
- **Recall@10**, because a re-ranker can raise nDCG by reordering the same relevant passages without finding any more, and this separates the two.
- **MRR@10**, because a search user reads from the top and the rank of the first relevant passage is what they feel.
- **The paired nDCG@10 difference against BM25, with a 95% t interval.** Every method re-ranks the same candidates for the same queries, so the difference is paired per query. A mean is a result only when the interval stays on one side of zero. This is the number the experiment turns on.
- **Latency p50 and p95 per call**, to price the stage, with the hardware caveat below.
- **Cost**, taken from the provider's own per-call figure, never tokens times a rate from a pricing page.
- **Call accounting**: retries, failures, and how many scored passages sit in a tie, because a tie is BM25's judgement counted inside a re-ranker's score.

## Assumptions and limits

- **On 91 of the 323 queries, no re-ranker could have changed anything.** BM25 retrieved 20 candidates per query, and for 91 of them not one is judged relevant by the official qrels. nDCG@10 is zero for BM25 and zero for every re-ranker on those queries whatever order they choose, so they contribute nothing but denominator. Re-ranking was possible on 232 queries, and the means are reported over all 323.
- **Laya and the cross-encoder run on CPU. There is no usable GPU on this machine**, so their latencies are CPU latencies and are not comparable to Laya's published 39.5 ms on a T4. Jev runs over the network, so its latency is a round trip and not a comparable quantity at all. Nothing here is a speed claim.
- **Jev is never compared with the cross-encoder.** Each interval is against the common BM25 floor, and two intervals that both exclude zero do not establish that one method beats the other. The paired difference would, and it is not measured.
- **The pilot and the full run are not directly comparable**, and where they disagree the full run is the one to believe. The pilot used top-50 candidates and the full run top-20, and a shallower pool is a cleaner one, which moves BM25's own floor with it.
- **A tie is not an opinion.** `rank_by_score` leaves tied passages in the order BM25 gave them, and Jev's answers come back rounded to two decimals, so a 0-to-4 score has at most 401 places to land. Coarse steps mean ties, and the rounding is not something a client can switch off.
- **Passages are cut to 1,000 characters.** NFCorpus abstracts are longer than the English checkpoint's roughly 320-token state budget, so they are cut where it is visible in the recorded request. Nothing here measures what the tail would have added.
- **A failed call keeps BM25's position.** Nothing is invented for a call that never succeeded, which means those passages are BM25's judgement counted inside the re-ranker's score.
- **The Laya passes and the Jev pass overlap in time.** Jev is network-bound and Laya is CPU-bound so they do not contend, but the Laya latencies are slightly pessimistic. They remain comparable to each other, which is what the base-versus-fine-tune question needs.

## How to run it

Needs [uv](https://docs.astral.sh/uv/) and Python 3.12.

### Laya's weights

Laya needs no key. About 2.3 GB download on first use into `models/`, which is not committed. If you already have them, point at that folder and nothing is downloaded:

```
export LAYA_PATH=/path/to/laya        # or pass --laya-path
```

If the weights are missing **and** the download fails, the run stops and says both of those things rather than half-loading a model.

### Jev's key

Jev is hosted and needs a key. Copy `.env.example` to `.env` at this repository's root:

```
OPENROUTER_API_KEY=your_key       # https://openrouter.ai/keys
```

Jev answers at `POST /api/v1/systemone`, not `chat/completions` — it is a decision model, so the typed question itself goes on the wire. The build is pinned to `typesafe/jev-1.13-20260917`, because `typesafe/jev-1.13` is a floating minor and `~typesafe/jev-latest` is an alias, and a benchmark that cannot say which build produced a number is not a benchmark.

`TYPESAFE_API_KEY` reaches the same model through TypeSafe's own API and wins when both are set. Either variable also works straight from the environment. Nothing but `jev-*` needs a key: BM25, the cross-encoder and both Laya checkpoints run entirely locally.

### The commands

A small pass first, to check the instrument:

```
uv sync
uv run python -m rerank.evaluate --limit 30 --top-k 50
```

The full run:

```
uv run python -m rerank.evaluate --limit 0 --top-k 20 \
    --methods bm25 laya-score laya-typed-score jev-score cross-encoder
```

`--limit 0` runs all 323 test queries. `--top-k` sets candidates per query. `--methods` picks from `bm25`, `laya-score`, `laya-noul`, `laya-typed-score`, `laya-typed-noul`, `jev-score`, `jev-noul`, `cross-encoder`.

The full run is 6,460 scoring calls per re-ranker, 25,840 in all, and took about 3 h 40 m of wall clock. Money is not the constraint — the whole Jev pass cost about fifteen cents — but the local passes are CPU-bound and the network pass retries, so budget an evening. What it actually cost and how long it actually took are in [RESULTS.md](RESULTS.md).

The tests:

```
uv run pytest
```

## Amendments

- **2026-09-23**, between the pilot and the full run: the full run carries the `score` formulation only and drops the `noul` and `boolean` variants. The pilot found neither formulation better than the other for either model, and repeating them at 323 queries would have doubled the cost of a question already answered. The full run also moves from top-50 candidates to top-20.
- **2026-09-25**, after the run: Jev is now reached over OpenRouter at `POST /api/v1/systemone`, with the build pinned to a date. The recorded run used the Vercel AI Gateway, whose key no longer exists; [RESULTS.md](RESULTS.md) records that as the transport for those numbers. Nothing was re-measured, so a re-run over OpenRouter is not guaranteed to reproduce the recorded latencies or costs.

# e-means vs k-means: experiment design

Status: draft for review, 2026-11-13.

## Intent

e-means (Bessa et al., arXiv 2611.09501, shipped in `faiss.contrib.e_means`, faiss-cpu ≥ 1.16) claims to reach Lloyd's 25-epoch k-means test error in 1.2–3 epochs, 8–15× faster on CPU. Its update rule is a moving average, which was first used for data that keeps changing, yet the paper tests only fixed datasets.

The experiment has three parts:

- **A. The race (base).** Does the claim hold on a laptop? We give each method the same wall-clock budget, measure the index it builds, and measure epochs and time to match Lloyd.
- **B. The growing index (headline).** New documents arrive every month. Does warm-started e-means keep a vector-search index's recall up for a fraction of the cost of rebuilding it, and is any gain from e-means itself or just from warm-starting?
- **C. The dimension sweep (add-on).** With one embedding model cut to 64…768 dimensions, does e-means' advantage change with dimension?

The result becomes a story page on lab.codewithnk.com/emeans, like `/rerank`.

## Goals, non-goals, anti-goals

- **Goals:** a measured answer to each of A, B and C, with intervals and a prediction recorded first. B is the lead of the write-up.
- **Non-goals:** GPU timing (their fast kernels need A100+/H100; a Kaggle T4 would time a fallback); reimplementing e-means; tuning any arm beyond its defaults; their toy local-minima demos.
- **Anti-goals:**
  - a page that restates their headline;
  - comparing a tuned arm with an untuned one;
  - a growing-index result where e-means "wins" only because it was the only arm allowed to warm-start;
  - timing arms on different thread counts.

## Approaches considered

| | Approach | Verdict |
|---|---|---|
| Hardware | Laptop CPU for all clustering and search. Kaggle GPU only to embed the corpus once (not timed) | Picked. A GPU timing on a T4 would be unfair to e-means |
| Second static dataset | Our own arXiv embeddings (picked) vs GIST1M | arXiv embeddings serve A, B and C at once. GIST would add a dataset used by nothing else. Note that VIBE's arXiv set is in their paper, but with a different embedding model and no time order |
| Growing-index updates | Warm-started e-means through its public `begin`/`accumulate`/`update` loop (picked) vs reimplementing the moving-average rule | Using their code keeps the test about their method |
| Fixed-budget framing | Fixed wall-clock budgets (picked) vs only "epochs to match" | Budgets are what a practitioner chooses; epochs-to-match is still reported |

## Data

1. **SIFT1M** (1M × 128, L2, ann-benchmarks HDF5 with 10k queries and true neighbours). The paper reports on SIFT, so it is the reproduction check. Used in A.
2. **arXiv-CS** (built here). Cornell arXiv metadata (Kaggle, CC0), `cs.*` primary category, January 2018 to December 2025, uniform sample of up to 10,000 abstracts per month (about 960k docs). Each abstract is embedded with `nomic-embed-text-v1.5` (768-d, Matryoshka, prefix `search_document:`, 8,192-token limit not reached) on a Kaggle GPU once. The vectors are stored float32, unit-normalised (so L2 k-means equals cosine), with publish month. Used in A (all 768-d, months shuffled), B (in time order) and C (truncated to 64/128/256/512/768 then re-normalised). The embedding model revision, sampling seed and per-month counts are recorded in the dataset card (`DATASET.md`).

k = 4,096 throughout (a usual IVF size at about 1M vectors).

## A. The race

**Split:** one fixed split (seed 0) per dataset. Hold out 100k vectors as the MSE test set and train on the rest. For SIFT, the IVF index holds all base vectors, so the ann-benchmarks ground truth stays valid. For arXiv, the 10k queries are drawn from the 100k test set. The index holds every doc except those queries, and we compute exact top-10 neighbours ourselves.

**Arms** (all CPU, float32 in memory; threads = physical cores, set the same way for every arm and recorded):
1. **Lloyd**: `faiss.Kmeans`, `max_points_per_centroid` raised so it trains on every training vector, seeded random init.
2. **e-means**: `NumpyEmeans`, defaults (`batch_law="auto"`).
3. **MiniBatchKMeans** (scikit-learn), `init="random"`, `n_init=1`, default batch size and reassignment, and early stopping off (`max_no_improvement=None`, `tol=0`), so `max_iter` counts full epochs. Threads are pinned with `threadpoolctl`. This control tells us whether the gain is e-means' update rule or just mini-batches.

**Two views of the same runs:**
- **Epochs to match.** Lloyd's test-MSE curve at iterations {1, 2, 3, 5, 10, 25} comes from one run per seed, snapshotting centroids. Lloyd's timing comes from a separate, uninterrupted 25-iteration run. e-means and MiniBatch run at budgets {1, 2, 3, 5} epochs, each its own run (e-means' step schedule depends on the budget). "Matches" means the mean paired MSE ratio over seeds is ≤ 1.000. If no budget up to 5 matches, the result is "did not match within 5 epochs", with no speedup.
- **Fixed wall-clock budgets: 10 s, 30 s, 60 s, 120 s.** For each arm and budget, the setting is the largest iteration or epoch count whose pilot-measured training time fits the budget. We report the actual time next to the budget, and any run that overshoots by more than 10% is flagged.

5 seeds per arm × setting. Seeds vary only initialisation and batch order.

**Measures:**
- Test MSE ratio to Lloyd-25, paired by seed.
- Speedup at match.
- Dead clusters (< 10 points) and imbalance factor.
- Search: an `IndexIVFFlat` per centroid set (centroids used as-is, all vectors added), at `nprobe ∈ {1, 4, 16, 64}`:
  - recall@10;
  - vectors scanned per query, mean and p95;
  - queries per second (batch search after one warm-up pass).

## B. The growing index

**Timeline:** arXiv-CS in publish order. The **initial index** is trained on 2018–2019 (about 240k docs). Then the 72 months from 2020-01 to 2025-12 arrive one at a time. Each month, 500 of that month's docs are held out as **queries** and never indexed. The rest are added to the index.

**Arms**, all starting from the same 2018–2019 Lloyd-25 centroids for a given seed:
1. **Stale**: centroids never change, and new docs are just added.
2. **Yearly rebuild**: every January, a Lloyd-25 retrain from scratch on all indexed docs, then all docs are re-assigned. This is common practice.
3. **Monthly e-means**: each month, warm-start `NumpyEmeans` from the current centroids via `begin(n, init_centroids)`, run **0.5 epoch** (fractional budgets are supported) over all indexed docs in a seeded shuffle, `finalize`, then re-assign all docs.
4. **Monthly Lloyd, same rows**: each month, one Lloyd iteration warm-started from the current centroids, over a seeded random half of the indexed docs (the same number of rows that arm 3 touches), then re-assign all docs. This control separates "e-means" from "warm-starting".

**Why a half epoch over the whole index, not a small window.** `finalize` replaces every assigned centroid with the plain mean of its last-pass points, and `begin` starts each cluster's moving-average mass near zero. On a small window (one month plus a little history: about 5 points per centroid), arms 3 and 4 would both collapse to "window means" and forget the older docs. A half pass over the whole index gives about 29 points per centroid at the start of the timeline and about 117 at the end. The finalize behaviour belongs to e-means, so we keep it and describe it in the write-up.

**The compute ledger counts the same things for every arm:**
- Maintenance CPU = training plus re-assignment.
- Adding new docs costs the same in every arm, so it is reported separately and left out of comparisons.
- Stale's maintenance cost is 0.
- Arms 3 and 4 touch the same rows by construction, so their measured times are compared directly with no matching step.
- Rough cost per year: monthly = 12 × (0.5 + 1) = 18 passes over the index; yearly = 25 + 1 = 26.

**Queries:** each month's 500 held-out docs are never indexed and never appear in any training pass, rebuild or initial index.

**Measured each month, per arm:**
- recall@10 for that month's queries, against exact top-10 over everything indexed so far, at `nprobe` 16 and 64;
- vectors scanned per query (mean, p95);
- imbalance;
- dead clusters;
- maintenance CPU seconds that month, plus cumulative.

**Seeds:** 3. A seed sets the initial index's Lloyd init (shared by all four arms) and the shuffle and sample order. Each month's query set is fixed.

**How the claims are judged:** paired per-query bootstrap over 2025's 6,000 queries × 3 seeds for the end-state comparisons. The month-by-month curves show mean and range across seeds.

## C. The dimension sweep

arXiv-CS (shuffled, the static split from A), truncated to d ∈ {64, 128, 256, 512, 768} and re-normalised. Lloyd vs e-means only. We report epochs to match and speedup at match, as in A. 3 seeds per dimension.

## What we expect (prediction, recorded before any run)

- **A.** On SIFT1M, e-means matches Lloyd-25 within 3 epochs, 5–15× faster on CPU. On arXiv-CS-768 it matches within 5 epochs. MiniBatch matches on neither within 5 epochs. At the 10 s and 30 s budgets, e-means' index has higher recall@10 at nprobe 16 than Lloyd's; at 120 s the two are within 0.5 points. Wherever recall is equal, e-means scans fewer vectors at p95 (more even clusters).
- **B.**
  - Stale's 2025 recall@10 at nprobe 16 is at least 2 points below yearly rebuild's, with the gap opening fastest from 2023 (the LLM paper surge).
  - Yearly rebuild shows a sawtooth.
  - Monthly e-means' 2025 recall is within 0.5 points of yearly rebuild's, at lower cumulative maintenance CPU (about 0.7× by the pass count above; measured).
  - Monthly Lloyd with the same rows ends within 0.5 points of monthly e-means. In that case the win belongs to warm-starting, not to e-means, and the write-up says so.
- **C.** Epochs-to-match is flat across dimension (within one budget step). Speedup at match varies by less than ±30% across dimension.

## Protocol and engineering (per running-experiments)

- Folder `emeans/` mirroring `rerank/`. README (the protocol, committed before the first run) and RESULTS (after), harness modules, `tests/`, committed `results/`, gitignored `runs/`, and `figures.py` generating every chart from `results/`.
- One JSONL record per unit of work, keyed and resumable: (part, dataset, arm, setting, seed) for A and C, plus (arm, seed, month) for B. Each record holds the centroids' hash, MSE, timings, imbalance, dead clusters, and per-nprobe recall and scanned counts.
- We record faiss, scikit-learn, numpy and sentence-transformers versions, the embedding model revision, CPU model, thread count, commit and date. Timing covers training (and re-assignment in B) only, via `perf_counter`.
- **Pilot first:** SIFT1M 100k subset with k = 256 and one seed, plus three months of the B timeline. The pilot checks that:
  - Lloyd's train MSE never goes up;
  - nprobe = k gives recall 1.0;
  - records are complete;
  - warm-started e-means runs through `begin`.

  It then projects full-run wall clock for each part. Full runs start only after the projection.
- Registered with `cli`, so `uv run cli run emeans --part a|b|c --tag full` works. The embedding job is a Kaggle notebook under `emeans/kaggle/`, using the existing `cli submit` path.

## Write-up

A site page `/emeans` and blog-style story, leading with B:
1. the problem of an index that goes stale;
2. what k-means does inside vector search (one picture);
3. the growing-index result;
4. whether the speed claim reproduced (A, as the race chart);
5. whether dimension matters (C);
6. what this doesn't show.

Every figure is generated from `results/`. It's built after the results, as its own slice.

## Left to the builder

- Module split and exact JSONL schema.
- How Lloyd snapshots are taken.
- Where downloads and the embedding cache are stored.
- The arXiv sampling code.
- Figure styling within the site's conventions.

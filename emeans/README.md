# Does e-means keep a growing vector index fresh more cheaply than rebuilding it?

Results: [RESULTS.md](RESULTS.md), once there are some. How far this got is
read off `results/` by `uv run cli list`, not written down here.

## The question

e-means (Bessa et al., arXiv 2611.09501, shipped in `faiss.contrib.e_means`) claims to reach Lloyd's 25-epoch k-means test error in 1.2 to 3 epochs, 8 to 15 times faster on CPU. This experiment asks three things, in order of importance:

- **B (headline).** New documents arrive every month. Does warm-started e-means keep an IVF vector-search index's recall up for a fraction of the cost of rebuilding it, and is any gain from e-means itself or just from warm-starting?
- **A (base).** Does the speed claim reproduce on a laptop CPU, with the same wall-clock budget for every method?
- **C (add-on).** With one embedding model cut to 64 to 768 dimensions, does e-means' advantage change with dimension?

## Motivation

An IVF index is only as good as its centroids. Documents change over time, so the centroids go stale and recall drops, and the usual fix is a periodic full rebuild. e-means' update rule is a moving average, which is the kind of rule made for data that keeps changing, yet the paper tests only fixed datasets. If a cheap monthly warm-started update holds recall near a yearly rebuild's, that changes how an index is maintained. If a plain warm-started Lloyd step does just as well, the credit belongs to warm-starting, and that is worth knowing too.

## What we expect

Written before any run.

- **A.** On SIFT1M, e-means matches Lloyd-25 within 3 epochs, 5–15× faster on CPU. On arXiv-CS-768 it matches within 5 epochs. MiniBatch matches on neither within 5 epochs. At the 10 s and 30 s budgets, e-means' index has higher recall@10 at nprobe 16 than Lloyd's; at 120 s the two are within 0.5 points. Wherever recall is equal, e-means scans fewer vectors at p95 (more even clusters).
- **B.**
  - Stale's 2025 recall@10 at nprobe 16 is at least 2 points below yearly rebuild's, with the gap opening fastest from 2023 (the LLM paper surge).
  - Yearly rebuild shows a sawtooth.
  - Monthly e-means' 2025 recall is within 0.5 points of yearly rebuild's, at lower cumulative maintenance CPU (about 0.7× by the pass count above; measured).
  - Monthly Lloyd with the same rows ends within 0.5 points of monthly e-means. In that case the win belongs to warm-starting, not to e-means, and the write-up says so.
- **C.** Epochs-to-match is flat across dimension (within one budget step). Speedup at match varies by less than ±30% across dimension.

What would falsify it: e-means not matching Lloyd-25 within 5 epochs (A); monthly e-means more than 0.5 points below yearly rebuild at 2025, or costing more CPU than it (B); epochs-to-match moving by more than one budget step across dimension (C).

## Data

1. **SIFT1M**: 1M x 128, L2, the ann-benchmarks file `sift-128-euclidean.hdf5` (10k queries with true neighbours). The paper reports on SIFT, so it is the reproduction check. Used in A. Download it from the ann-benchmarks site and put it at `<cache-dir>/sift-128-euclidean.hdf5` (default cache dir `emeans/runs/data`).
2. **arXiv-CS**, built here: Cornell arXiv metadata (Kaggle, CC0), `cs.*` primary category, January 2018 to December 2025, a uniform sample of up to 10,000 abstracts per month. Embedded once with `nomic-embed-text-v1.5` (768-d, prefix `search_document:`) on a Kaggle GPU, not timed. Stored float32, unit-normalised (so L2 k-means equals cosine), with publish month. Used in A (months shuffled), B (in time order) and C (truncated to d in 64, 128, 256, 512, 768 and re-normalised). Model revision, sampling seed and per-month counts go in [DATASET.md](DATASET.md). The spec's "about 960k docs" and "about 240k initial docs" assume the cap is reached every month; real monthly volumes are lower in the early years, so the real counts in `DATASET.md` replace those figures, and every "points per centroid" figure below is approximate.

k = 4,096 for every part.

**Split** (seed 0, one fixed split per dataset): 100k vectors are held out as the MSE test set and the rest is the training set. SIFT: the IVF index holds all base vectors, so the shipped ground truth stays valid. arXiv: 10k queries are drawn from the 100k test set, the index holds every document except those queries, and we compute exact top-10 neighbours ourselves.

## Method

### Setup common to every arm

- All timing is on the laptop CPU, float32, threads = physical cores, the same for every arm. The thread count and the thread pools actually observed are recorded. Timing covers training (plus re-assignment in B) only, via `perf_counter`.
- Seeds vary only initialisation and batch or shuffle order: 5 seeds per arm and setting in A, 3 in B and C.
- No arm is tuned. Defaults only, and nothing is ever chosen on the test set.

### A. The race

Arms:

1. **Lloyd**: `faiss.Kmeans`, `max_points_per_centroid` raised to cover every training vector, seeded random init.
2. **e-means**: `NumpyEmeans` with defaults (`batch_law="auto"`).
3. **MiniBatchKMeans** (scikit-learn): `n_init=1`, default batch size and reassignment, `max_no_improvement=None`, `tol=0`, so `max_iter` counts full epochs; threads pinned with `threadpoolctl`. This control tells us whether a gain is e-means' update rule or just mini-batches.

Two views of the same runs:

- **Epochs to match.** Lloyd's test MSE at iterations {1, 2, 3, 5, 10, 25} comes from one snapshotting run per seed; Lloyd's timing comes from a separate uninterrupted 25-iteration run. e-means and MiniBatch run at {1, 2, 3, 5} epochs, each its own run (e-means' step schedule depends on the budget). "Matches" means the mean paired MSE ratio over seeds is at most 1.000. If no setting up to 5 epochs matches, the result is "did not match within 5 epochs" and no speedup is reported.
- **Fixed wall-clock budgets of 10 s, 30 s, 60 s and 120 s.** For each arm and budget, the setting is the largest iteration or epoch count whose measured training time fits the budget. The actual time is reported next to the budget and any run that overshoots by more than 10% is flagged, never clipped.

### B. The growing index

Timeline: arXiv-CS in publish order. The initial index is the 2018-2019 Lloyd-25 centroids. The 72 months from 2020-01 to 2025-12 then arrive one at a time. Each month 500 of that month's docs are held out as queries and never indexed, never in any training pass, rebuild or the initial index. The rest are added.

Arms, all starting from the same initial centroids for a given seed:

1. **Stale**: centroids never change; new docs are added.
2. **Yearly rebuild**: every January (2020-01 included, so six rebuilds), a Lloyd-25 retrain from scratch on all indexed docs, then all docs re-assigned.
3. **Monthly e-means**: each month, warm-start `NumpyEmeans` from the current centroids via `begin(n, init_centroids)`, run 0.5 epoch over all indexed docs in a seeded shuffle, `finalize`, then re-assign all docs.
4. **Monthly Lloyd, same rows**: each month, one Lloyd iteration warm-started from the current centroids over a seeded random half of the indexed docs (the same number of rows arm 3 touches), then re-assign all docs. This separates "e-means" from "warm-starting".

Why a half epoch over the whole index and not a small window: `finalize` replaces every assigned centroid with the plain mean of its last-pass points and `begin` starts each cluster's moving-average mass near zero. On a window of about 5 points per centroid, arms 3 and 4 would both collapse to window means and forget older docs. A half pass over the whole index gives about 29 points per centroid at the start and about 117 at the end (upper bounds if the monthly cap is not reached).

Every month, for every arm, the order is maintain then measure. The compute ledger counts the same things for every arm:

- Maintenance CPU = training plus re-assignment of all indexed docs. Re-assigning all docs necessarily includes the newest ones, and that is inside maintenance.
- Adding the new docs to the index costs the same in every arm, so it is reported separately and left out of comparisons.
- Stale's maintenance is 0.
- Arms 3 and 4 touch the same rows by construction, so their measured times are compared directly.
- Rough cost per year: monthly = 12 x (0.5 + 1) = 18 passes over the index; yearly = 25 + 1 = 26.

Per month and arm: recall@10 against exact top-10 over everything indexed so far at `nprobe` 16 and 64 (per-query values kept, for the paired bootstrap), vectors scanned (mean, p95), imbalance, dead clusters, maintenance CPU seconds that month and cumulative.

End-state claims use a paired per-query bootstrap over 2025's 6,000 queries (12 x 500), the per-query difference averaged over the 3 seeds, with the three per-seed mean differences reported alongside. Month curves show mean and range across seeds.

### C. The dimension sweep

arXiv-CS (shuffled, the static split from A) truncated to d in {64, 128, 256, 512, 768} and re-normalised. Lloyd vs e-means only; epochs to match and speedup at match as in A. 3 seeds per dimension. No queries or search are needed.

### Decisions made while planning

Dated 2026-11-13, before any run. Where the spec left room, these fix the choice.

1. **Budgets come from a full-scale calibration, not the pilot.** The spec says the budget setting is the largest count whose pilot-measured time fits; a 100k, k=256 pilot cannot predict full-scale seconds per epoch. So "pilot-measured" means a one-epoch calibration run per arm at full scale, stored as records, assuming time is linear in epochs. The pilot itself checks correctness and projects wall clock.
2. **Same init for all arms.** Each seed's random initial centroids feed Lloyd, e-means and MiniBatch (MiniBatch gets them as an array rather than `init="random"`). This pairs the arms per seed; e-means' own default init is also random rows.
3. **Search is evaluated only for `iters=25` and the budget settings**, the indexes a practitioner would build, not for every epoch setting, to bound cost.
4. **Lloyd snapshots are a warm-started chain.** `faiss.Kmeans` has no per-iteration hook, so the curve at {1, 2, 3, 5, 10, 25} is taken by continuing from the previous snapshot. It is not bit-identical to one uninterrupted run (empty-cluster splitting restarts its random draws per call); the uninterrupted run is the timing source.
5. **Publish month** is the month of the first version's `created` date; primary category is the first listed category.
6. **Yearly rebuild includes 2020-01**, and every month is maintain-then-measure for all arms.
7. **End-state bootstrap** resamples the 6,000 queries, the per-query difference averaged over seeds.
8. **Per-query recall is stored in the B records**, which the paired bootstrap needs.

### Pilot

Before any full run: SIFT1M first 100k rows, k = 256, one seed, plus three months of the B timeline. It checks that Lloyd's train MSE never rises, that `nprobe = k` gives recall 1.0, that records are complete, and that warm-started e-means runs through `begin`; then it projects the full-run wall clock per part. Full runs start only after the projection is approved.

`--tag pilot` selects the pilot scale; `--tag full` selects the full scale. Any other tag is refused, so a typo cannot start a multi-hour run.

## Metrics

- **Test MSE ratio to Lloyd-25**, paired by seed: the quality of the centroids as a quantiser, held out from training.
- **Epochs to match and speedup at match**: the reproduction of the paper's claim.
- **Dead clusters** (under 10 points) and **imbalance** (`k * sum(n_i^2) / n^2`, 1.0 is even): what an uneven partition costs a search.
- **recall@10, vectors scanned per query (mean, p95), queries per second** at `nprobe` in {1, 4, 16, 64}: what a practitioner gets from the index. Scanned counts matter because equal recall with fewer scans is a real gain.
- **Maintenance CPU seconds** (B): the cost side of the headline.

## Assumptions and limits

- CPU only. e-means' fast kernels target A100/H100 class GPUs; a Kaggle T4 would time a fallback, so no GPU timing is attempted.
- One laptop, one machine. Absolute seconds do not transfer; ratios between arms measured under identical conditions are the claim.
- Embeddings come from one model on one corpus (arXiv-CS abstracts). Dimension truncation is raw truncate plus re-normalise, as the spec says, not the model card's layer-norm-first recipe.
- OpenMP reductions are not bit-identical across runs, so "arms start identical" means equal to numerical noise, and re-runs reproduce records except for timing fields.
- Arms are not tuned. A tuned arm would be a different experiment.

## How to run it

Run the commands from the repository root after `uv sync`.

```
uv run cli list
uv run cli run emeans --part a --tag pilot --dry-run   # list the units of work and stop
uv run cli run emeans --part a --tag pilot             # pilot scale (SIFT1M 100k subset, k = 256)
uv run cli run emeans --part a --tag full              # full scale; hours of machine time
```

`--part` is `a`, `b` or `c` (default `a`); only part A exists so far. `--tag` is `pilot` or `full`, nothing else. `--cache-dir` is where `sift-128-euclidean.hdf5` (from ann-benchmarks.com) lives; `--data-dir` is where the arXiv vectors live (default `emeans/runs/data`). Part A appends one record per unit of work to `runs/<tag>/a.jsonl` and is resumable: a rerun skips every finished unit and redoes a half-written one. The environment, with the thread pools actually observed, goes to `runs/<tag>/environment.json`. `--dry-run` lists the units and which are done, and reads no data.

The pilot is a human gate: run it on the laptop, read `results/pilot/projection.json` (hours per part, linear in n * k * d * epochs; `a_arxiv` is extrapolated from SIFT, `b` and `c` are null until those parts exist), and commit `results/pilot/`. The command exits non-zero and prints each failed check if the pilot's records are wrong. Do not start a `full` run before the projection is approved.

This section is updated as parts land; the protocol sections above are not edited once a run has started.

## Protocol amendments

**Amendment 2026-11-13 (E2, before any run).**

- Recall@10 is counted by distance, not id: a returned neighbour counts when its distance is within float tolerance of the true 10th-neighbour distance. SIFT1M holds duplicate vectors, so exact and IVF search order tied neighbours differently and an id match would score correct answers as misses. This applies to every recall figure in every part.
- The pilot also searches at `nprobe = k` (256), where recall must be 1.0; the full runs use the nprobe set above only.
- The pilot runs the calibration, snapshot, Lloyd 25-iteration and epoch units, and leaves the fixed-budget units out: budgets are derived from a full-scale calibration, which a 100k, k = 256 pilot cannot give. The projection adds the budget units' fixed time.
- Imbalance and dead clusters are measured on the labels of the indexed vectors, for every trained setting.

# Running this experiment on a Kaggle GPU

This machine has no usable GPU, so every local model here has been measured on
CPU and `RESULTS.md` says plainly that nothing in it is a speed claim. A Kaggle
notebook has a T4 for free, which is enough to measure every method on one
machine and one device in a single run.

## Once

1. **A Kaggle API token.** kaggle.com -> Settings -> API -> *Create New API
   Token*. Put the downloaded `kaggle.json` at `~/.kaggle/kaggle.json`
   (`C:\Users\<you>\.kaggle\kaggle.json` on Windows).
2. **The OpenRouter key, as a Kaggle secret.** Open the notebook on kaggle.com
   -> *Add-ons* -> *Secrets*, add one named `OPENROUTER_API_KEY`, and attach it
   to this notebook. It is read by `UserSecretsClient` at run time and never
   written to a file, printed, or included in the output.
3. `pip install kaggle`, then put your Kaggle username in `kernel-metadata.json`.

## Every time

```
kaggle kernels push   -p rerank/kaggle                  # upload and start
kaggle kernels status <username>/rerank-nfcorpus-gpu    # poll
kaggle kernels output <username>/rerank-nfcorpus-gpu -p out/   # pull results back
```

The kernel clones this repository at `main`, so **push before you push**:
anything uncommitted is not what runs.

## What comes back

`out/` holds what the run wrote under `--out`:

| | |
|---|---|
| `results/gpu/` | `summary.json`, the CSV tables, and `figures/` |
| `runs/gpu/` | `scores.jsonl`, the raw wire log, and the pinned candidate set |

Both, deliberately. The wire log is the evidence the results are derived from,
and without it a number can never be recomputed without paying for the run
again.

## Why the tag is `gpu` and not `full`

It is a different measurement, not a faster way to get the same one. `laya` runs
in **float16 on CUDA** and float32 on CPU -- `amp_enabled` is unconditional on
CUDA, and a T4's compute capability of 7.5 forces fp16 -- so its answers are not
the ones the CPU produced. Ties are what this experiment is sensitive to, and
rounding moves ties. The results carry `device: cuda` on every row so the two
can never be quietly averaged.

Leave laya's `fast=True` TileLang path alone for the same reason: it is another
GPU-only variable and would confound the comparison.

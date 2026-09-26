# Running this experiment on a Kaggle GPU

## Once

1. **Kaggle API token.** kaggle.com -> Settings -> API -> *Create New API Token*.
   Put `kaggle.json` at `~/.kaggle/kaggle.json`
   (`C:\Users\<you>\.kaggle\kaggle.json` on Windows), then `pip install kaggle`.
2. **The OpenRouter key as a Kaggle secret.** On the notebook page: *Add-ons* ->
   *Secrets*, add `OPENROUTER_API_KEY` and attach it. It is read at run time and
   never written to a file, printed, or included in the output.
3. Put your Kaggle username in the `id` field of `kernel-metadata.json`.

## Every time

```
kaggle kernels push   -p rerank/kaggle
kaggle kernels status <username>/rerank-nfcorpus-gpu
kaggle kernels output <username>/rerank-nfcorpus-gpu -p out/
```

The kernel clones this repository at `main`, so push your commits first --
anything uncommitted is not what runs.

## What comes back

| | |
|---|---|
| `out/results/gpu/` | `summary.json`, the CSV tables, `figures/` |
| `out/runs/gpu/` | `scores.jsonl` and the pinned candidate set |

Both: the wire log is what the results are derived from, and without it a number
cannot be recomputed without paying for the run again.

## The tag is `gpu`

`laya` runs float16 on CUDA and float32 on CPU, so its answers differ between
them. `usable_device()` records `cuda` on every row, and the two tags are never
merged. Leave laya's `fast=True` TileLang path off: it is another GPU-only
variable.

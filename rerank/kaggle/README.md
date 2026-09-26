# Running this experiment on a Kaggle GPU

## Once

1. **Authenticate the CLI.** `uv run kaggle config view` should print your
   username. If it does not, `uv run kaggle auth login` opens a browser and
   writes `~/.kaggle/access_token`. Do not also create a `kaggle.json`: it is a
   second auth path and it shadows the token.
2. **The OpenRouter key as a Kaggle secret.** On the notebook page: *Add-ons* ->
   *Secrets*, add `AI_EXPERIMENTS_OPENROUTER` and attach it. The kernel reads it
   into `OPENROUTER_API_KEY`, which is where the reranker looks; it is never
   written to a file, printed, or included in the output.

   Its own name, and ideally its own OpenRouter key rather than one shared with
   this account's other notebooks: revoking or rotating one then does not touch
   the other, and this experiment's reported cost is only its own.

## Every time

```
uv run kaggle kernels push   -p rerank/kaggle
uv run kaggle kernels status nadeem4nk/rerank-nfcorpus-gpu
uv run kaggle kernels output nadeem4nk/rerank-nfcorpus-gpu -p out/
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

## Shares the account's GPU quota

The weekly GPU allowance is per account, not per project, so this draws from the
same budget as anything else running under `nadeem4nk`.

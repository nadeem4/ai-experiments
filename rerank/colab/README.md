# Running this experiment on a Colab GPU

[`run.ipynb`](run.ipynb) runs the same command the Kaggle kernel runs. Open it in
Colab ([badge link](https://colab.research.google.com/github/nadeem4/ai-experiments/blob/main/rerank/colab/run.ipynb))
and run every cell.

## Once

1. **Runtime -> Change runtime type -> T4 GPU.**
2. **Secrets** (the key icon in the sidebar) -> add `OPENROUTER_API_KEY` and give
   the notebook access. It is read at run time and never printed.

## What it does

Clones this repository at `main` -- only `cli/` and `rerank/`, about 47 files of
195 -- installs the five packages Colab lacks, and runs:

```
python -m cli run rerank --tag gpu --out /content/out --limit 0 --top-k 20
```

Results land in `/content/out/results/gpu/` and the raw wire log in
`/content/out/runs/gpu/`. The last cell zips both and downloads them.

Push your commits first: the notebook clones `main`, so anything uncommitted is
not what runs.

## Colab or Kaggle

Kaggle is scriptable end to end -- `uv run cli submit rerank` pushes and starts
it, and it needs no browser. Colab is not: the session is attached to a tab.

Kaggle's GPUs are free and shared, so a run can sit queued for an hour with
nothing wrong. Colab Pro is paid, so the GPU is there when you ask. Use whichever
matters more that day.

Both produce the same `--tag gpu` results from the same commit, and `laya` runs
float16 on either.

# Embedding the arXiv-CS sample on a Kaggle GPU

Runs once, untimed. Everything timed in the experiment runs on the laptop CPU.

## Once

1. **Authenticate the CLI.** `uv run kaggle config view` should print your username; if not, `uv run kaggle auth login`.
2. **No secrets are needed.** The notebook attaches the public `Cornell-University/arxiv` dataset (CC0) and downloads `nomic-ai/nomic-embed-text-v1.5` from Hugging Face (internet on). The kernel installs `transformers<5` and `sentence-transformers<5`: the model's remote code (`nomic-bert-2048`) calls `get_extended_attention_mask`, which transformers 5 removed.
3. **GPU on.** In the notebook editor (`https://www.kaggle.com/code/nadeem4nk/emeans-arxiv-embed/edit`) -> Session options, pick a GPU accelerator. `uv run cli submit emeans` prints this reminder.

## Every time

The kernel clones this repository at `main`, so merge and push first; anything uncommitted is not what runs.

```
uv run cli submit emeans --dry-run
uv run cli submit emeans
uv run cli status emeans
uv run cli fetch emeans -p emeans/runs/data
uv run cli dataset emeans
```

## What comes back

| | |
|---|---|
| `vectors.npy` | float32 `(n, 768)`, unit norm, sorted by month then id |
| `months.npy` | publish month `YYYY-MM` per row |
| `ids.txt` | arXiv id per row |
| `meta.json` | model, the exact model commit used, dimensions |

The kernel resolves the model's commit hash at run time and records it in `meta.json`; `embed.REVISION` is `main` until that hash is copied into it. Read the per-month counts in `DATASET.md` before going on: the spec's "about 960k docs" is an estimate, and the real total sets the points-per-centroid figures.

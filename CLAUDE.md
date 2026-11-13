
# AI Experiments

Measured experiments on AI models, each keeping the exact request and response of every call; results are published at lab.codewithnk.com.

## Commands
- Setup: `uv sync`, and `npm install` in `site/`
- Test: `uv run pytest`, and `npm test` in `site/`
- Run: `uv run cli list` / `uv run cli run <name> --tag <tag>`; site: `npm run dev` in `site/`

## Layout
- `rerank/`: the re-ranking experiment. README.md is the protocol, RESULTS.md the findings, `experiment.py` and `evaluate.py` the harness, `results/` committed metrics, `runs/` raw records (gitignored)
- `emeans/`: the e-means vs k-means experiment (README is the protocol). `experiment.py` the `cli` entry, `race.py` parts A and C, `growing.py` part B, `stats.py`, `tables.py`, `pilot.py`, plus `store.py`, `data.py`, `metrics.py`, `index.py`, `env.py`, `arxiv.py`/`embed.py`, `kaggle/` the embedding notebook, `DATASET.md` the data card
- `cli/`: `uv run cli` entry. `__init__.py` holds the experiment registry, `kaggle.py` the Kaggle submit/fetch
- `site/`: Next.js site for lab.codewithnk.com. `experiments.config.ts` sets what is published, `app/<name>/` holds the pages, `data/` their JSON
- `tests/`: tests for the shared CLI and helpers
- `docs/superpowers/`: `specs/` design docs, `plans/` implementation plans
- `EXPERIMENT_TEMPLATE.md`: skeleton for a new experiment's README

## Conventions
- One experiment per top-level folder, self-contained. Commit the protocol (README) before the run and the results (RESULTS.md) after.

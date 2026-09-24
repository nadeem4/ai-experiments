"""Laya as a reranker: one typed question per (query, passage), zero-shot.

Two formulations are run on identical data, because nobody has published which
one ranks better and the second pass is cheap:

  score  an ordered list of rungs, irrelevant to directly-answers; rank by the
         expected position on that scale (0 to 4 here)
  noul   no criteria at all, one probability back; rank by that probability

Both run against two checkpoints out of one weights folder, loaded the same way:
the base English one, and the `typed-decisions` fine-tune that Convai's model
card says is where the capability on typed decisions actually comes from.

The weights (about 2.3 GB) are downloaded on first use into `models/laya`.
`LAYA_PATH`, or `--laya-path`, points at a folder you already have instead.
"""
import os
import time
from pathlib import Path

REPO = "convaiinnovations/laya"
DEFAULT_WEIGHTS_DIR = "models/laya"

SCORE_QUESTION = {
    "relevance": {
        "type": "score",
        "instructions": "A search engine returned this passage for this query. How relevant is the passage to the query?",
        "criteria": [
            "irrelevant: the passage is about something else entirely",
            "same general subject, but it does not address the query",
            "useful background on the query, but no answer",
            "partly answers the query",
            "directly answers the query",
        ],
    }
}

NOUL_QUESTION = {
    "relevance": {
        "type": "noul",
        "instructions": "A search engine returned this passage for this query. Does this passage answer the query?",
    }
}

# (questions, field, checkpoint). Laya ships three checkpoints; the base English
# one is `None`. Convai's own model card says the base checkpoints sit below the
# majority-class baseline on typed decisions and that the capability comes from
# fine-tuning, so `typed-decisions` is run on the identical candidates and the
# identical questions -- the checkpoint is the only thing that differs.
VARIANTS = {
    "laya-score": (SCORE_QUESTION, "score", None),
    "laya-noul": (NOUL_QUESTION, "noul", None),
    "laya-typed-score": (SCORE_QUESTION, "score", "typed-decisions"),
    "laya-typed-noul": (NOUL_QUESTION, "noul", "typed-decisions"),
}

# The English checkpoint reads 512 tokens, ~192 of them spent on the question, so
# the state gets roughly 320. Long NFCorpus abstracts are cut here, where it is
# visible in the recorded request, rather than silently inside the tokenizer.
MAX_PASSAGE_CHARS = 1000


def build_state(query, passage, max_chars=MAX_PASSAGE_CHARS):
    title, text = passage.get("title", "").strip(), passage.get("text", "").strip()
    body = f"{title}. {text}" if title else text
    return {"query": query, "passage": body[:max_chars]}


class LayaReranker:
    """`predict` is laya's Agent.predict, or anything with its signature."""

    def __init__(self, name, questions, field, predict):
        self.name, self.questions, self.field, self.predict = name, questions, field, predict

    def score(self, query, passage):
        state = build_state(query, passage)
        started = time.perf_counter()
        response = self.predict(state, self.questions)
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "score": float(response["answers"]["relevance"][self.field]),
            "request": {"state": state, "questions": self.questions},
            "response": response,
            "latency_ms": latency_ms,
        }


def resolve_weights(path=None, env=None):
    """An explicit path beats $LAYA_PATH, which beats this repo's models/laya."""
    env = os.environ if env is None else env
    return Path(path or env.get("LAYA_PATH") or DEFAULT_WEIGHTS_DIR)


def ensure_weights(path, download=None):
    """Downloads the weights on first use. Hugging Face's own cache uses symlinks,
    which fail on Windows without Developer Mode, so they go in a plain dir.

    No weights and no way to fetch them is the one case with no answer, so it
    names both escape hatches rather than raising out of huggingface_hub."""
    path = Path(path)
    if not (path / "model.safetensors").exists():
        if download is None:
            from huggingface_hub import snapshot_download as download
        try:
            download(repo_id=REPO, local_dir=str(path))
        except Exception as e:
            raise RuntimeError(
                f"Laya's weights are not in {path}, and downloading {REPO} failed: {e}. "
                f"Set LAYA_PATH (or pass --laya-path) to a folder holding model.safetensors."
            ) from e
    return path


def load(name, path=None, checkpoint=None, predict=None):
    """The variant picks the checkpoint; `checkpoint` overrides it by hand."""
    questions, field, default_checkpoint = VARIANTS[name]
    checkpoint = checkpoint or default_checkpoint
    if predict is None:
        path = resolve_weights(path)
        ensure_weights(path)
        import laya  # heavy import (torch); only needed when Laya actually runs
        agent = laya.load(path, subfolder=checkpoint) if checkpoint else laya.load(path)
        predict = agent.predict
    return LayaReranker(name, questions, field, predict)

"""Laya as a reranker: one typed question per (query, passage), zero-shot.

Two formulations are run on identical data, because nobody has published which
one ranks better and the second pass is cheap:

  score  an ordered list of rungs, irrelevant to directly-answers; rank by the
         expected position on that scale (0 to 4 here)
  noul   no criteria at all, one probability back; rank by that probability

The weights are the same English checkpoint arena uses (arena/models/laya), and
they are loaded the same way.
"""
import time
from pathlib import Path

REPO = "convaiinnovations/laya"

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

QUESTIONS = {"laya-score": (SCORE_QUESTION, "score"), "laya-noul": (NOUL_QUESTION, "noul")}

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


def ensure_weights(path, download=None):
    """Same folder arena downloads into. Hugging Face's own cache uses symlinks,
    which fail on Windows without Developer Mode, so weights go in a plain dir."""
    path = Path(path)
    if not (path / "model.safetensors").exists():
        if download is None:
            from huggingface_hub import snapshot_download as download
        download(repo_id=REPO, local_dir=str(path))


def load(name, path=None, checkpoint=None, predict=None):
    """Checkpoints: None (English, what step 1 uses), "multilingual", "typed-decisions"."""
    questions, field = QUESTIONS[name]
    if predict is None:
        path = path or "../arena/models/laya"
        ensure_weights(path)
        import laya  # heavy import (torch); only needed when Laya actually runs
        agent = laya.load(path, subfolder=checkpoint) if checkpoint else laya.load(path)
        predict = agent.predict
    return LayaReranker(name, questions, field, predict)

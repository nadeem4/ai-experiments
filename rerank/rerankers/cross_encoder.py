"""The established re-ranker to beat: a MiniLM cross-encoder trained on MS MARCO.

It is not a decision model -- it emits one unbounded logit and nothing else, which
is exactly the contrast this demo is about. There is no request to record beyond
the two strings it was handed.
"""
import time

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_PASSAGE_CHARS = 1000  # same cut as Laya, so both re-rank the same text


class CrossEncoderReranker:
    name = "cross-encoder"

    def __init__(self, predict, model=MODEL):
        self.predict, self.model_name = predict, model

    def score(self, query, passage):
        title, text = passage.get("title", "").strip(), passage.get("text", "").strip()
        body = (f"{title}. {text}" if title else text)[:MAX_PASSAGE_CHARS]
        started = time.perf_counter()
        logit = float(self.predict([(query, body)])[0])
        latency_ms = (time.perf_counter() - started) * 1000
        return {
            "score": logit,
            "request": {"model": self.model_name, "query": query, "passage": body},
            "response": {"logit": logit},
            "latency_ms": latency_ms,
        }


def load(model=MODEL, predict=None):
    if predict is None:
        from sentence_transformers import CrossEncoder
        predict = CrossEncoder(model).predict
    return CrossEncoderReranker(predict, model)

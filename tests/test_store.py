"""The resumable store. A run that dies at 90 minutes must resume, not restart."""
from rerank.store import append, load, scored_keys


def _rec(method, qid, doc, score):
    return {"method": method, "query_id": qid, "doc_id": doc, "score": score,
            "request": {"state": {}}, "response": {}, "latency_ms": 1.0}


def test_appends_and_reads_back(tmp_path):
    path = tmp_path / "scores.jsonl"
    append(path, _rec("laya-score", "q1", "d1", 0.5))
    append(path, _rec("laya-score", "q1", "d2", 0.7))
    got = load(path)
    assert [r["doc_id"] for r in got] == ["d1", "d2"]
    assert got[0]["request"] == {"state": {}}


def test_missing_file_is_empty(tmp_path):
    assert load(tmp_path / "nope.jsonl") == []
    assert scored_keys(tmp_path / "nope.jsonl") == set()


def test_keys_identify_what_is_already_done(tmp_path):
    path = tmp_path / "scores.jsonl"
    append(path, _rec("laya-score", "q1", "d1", 0.5))
    append(path, _rec("laya-noul", "q1", "d1", 0.5))
    assert scored_keys(path) == {("laya-score", "q1", "d1"), ("laya-noul", "q1", "d1")}


def test_a_half_written_last_line_is_dropped_not_fatal(tmp_path):
    """Killing the process mid-write leaves a truncated line. Losing that one
    record is fine; refusing to resume is not."""
    path = tmp_path / "scores.jsonl"
    append(path, _rec("laya-score", "q1", "d1", 0.5))
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"method": "laya-score", "query_id": "q1", "doc_i')
    assert [r["doc_id"] for r in load(path)] == ["d1"]
    assert scored_keys(path) == {("laya-score", "q1", "d1")}

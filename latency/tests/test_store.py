"""The append-and-flush JSONL log, which doubles as the resume point.

Same shape as `rerank/store.py` and `banking77/store.py`; only the key differs --
here a decision is one (model, task, example) -- which is why it is a separate
twenty lines rather than a shared abstraction.
"""
from latency import store


def test_key_is_the_model_the_task_and_the_example():
    record = {"model": "gpt", "task": "highway", "example_id": "seed-0:t1"}
    assert store.key(record) == ("gpt", "highway", "seed-0:t1")


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / "runs" / "calls.jsonl"
    store.append(path, {"model": "gpt", "task": "highway", "example_id": "a", "latency_ms": 1.5})
    store.append(path, {"model": "gpt", "task": "highway", "example_id": "b", "latency_ms": 2.5})
    assert [r["example_id"] for r in store.load(path)] == ["a", "b"]
    assert store.load(path)[1]["latency_ms"] == 2.5


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert store.load(tmp_path / "nothing.jsonl") == []


def test_a_half_written_final_line_is_dropped_not_raised(tmp_path):
    path = tmp_path / "calls.jsonl"
    store.append(path, {"model": "gpt", "task": "highway", "example_id": "a"})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"model": "gp')
    assert [r["example_id"] for r in store.load(path)] == ["a"]


def test_pending_keeps_only_the_work_that_is_not_already_recorded(tmp_path):
    path = tmp_path / "calls.jsonl"
    store.append(path, {"model": "gpt", "task": "highway", "example_id": "b"})
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    done = store.done_keys(path)
    assert [r["id"] for r in store.pending(rows, "gpt", "highway", done)] == ["a", "c"]


def test_the_same_example_under_another_model_is_still_pending(tmp_path):
    path = tmp_path / "calls.jsonl"
    store.append(path, {"model": "gpt", "task": "highway", "example_id": "a"})
    done = store.done_keys(path)
    assert [r["id"] for r in store.pending([{"id": "a"}], "claude", "highway", done)] == ["a"]

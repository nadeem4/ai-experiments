"""The append-and-flush JSONL log, which doubles as the resume point.

Same shape as `rerank/store.py` and `banking77/store.py`; only the key differs --
here a decision is one (model, task, example) -- which is why it is a separate
twenty lines rather than a shared abstraction.
"""
from typed_decisions import store


def test_key_is_the_model_the_task_the_example_and_the_pass():
    record = {"model": "gpt", "task": "highway", "example_id": "seed-0:t1", "repeat": 0}
    assert store.key(record) == ("gpt", "highway", "seed-0:t1", 0)


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


# --- the repeat pass, for the determinism metric -----------------------------


def test_the_key_carries_the_repeat_pass_so_a_second_answer_is_not_a_duplicate():
    """`--repeat` asks a subset again to measure determinism. Without the pass in
    the key the resume logic would see the example as already done and skip it, and
    there would never be a second answer to compare."""
    first = {"model": "gpt", "task": "highway", "example_id": "e", "repeat": 0}
    second = {"model": "gpt", "task": "highway", "example_id": "e", "repeat": 1}
    assert store.key(first) == ("gpt", "highway", "e", 0)
    assert store.key(second) == ("gpt", "highway", "e", 1)
    assert store.key(first) != store.key(second)


def test_a_record_written_before_the_repeat_flag_existed_reads_as_the_first_pass():
    """Older records carry no `repeat` field. They are pass 0, not a crash."""
    assert store.key({"model": "gpt", "task": "highway", "example_id": "e"}) == ("gpt", "highway", "e", 0)


def test_pending_is_per_pass_so_the_repeat_runs_even_though_the_first_pass_is_done(tmp_path):
    path = tmp_path / "calls.jsonl"
    store.append(path, {"model": "gpt", "task": "highway", "example_id": "a", "repeat": 0})
    done = store.done_keys(path)
    assert [r["id"] for r in store.pending([{"id": "a"}], "gpt", "highway", done)] == []
    assert [r["id"] for r in store.pending([{"id": "a"}], "gpt", "highway", done, repeat=1)] == ["a"]


def test_both_answers_are_kept_in_the_store_rather_than_the_second_overwriting(tmp_path):
    """The comparison needs both, and an audit log that overwrote the first answer
    could not be checked afterwards."""
    path = tmp_path / "calls.jsonl"
    store.append(path, {"model": "gpt", "task": "highway", "example_id": "a",
                        "repeat": 0, "choice": "IDLE"})
    store.append(path, {"model": "gpt", "task": "highway", "example_id": "a",
                        "repeat": 1, "choice": "SLOWER"})
    assert [(r["repeat"], r["choice"]) for r in store.load(path)] == [(0, "IDLE"), (1, "SLOWER")]

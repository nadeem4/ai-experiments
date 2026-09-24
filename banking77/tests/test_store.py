from banking77 import store


def test_key_is_the_arm_and_the_example():
    assert store.key({"arm": "A", "example_id": "test-7"}) == ("A", "test-7")


def test_append_then_load_round_trips(tmp_path):
    path = tmp_path / "runs" / "records.jsonl"
    store.append(path, {"arm": "A", "example_id": "test-0", "probabilities": {"a": 0.5}})
    store.append(path, {"arm": "A", "example_id": "test-1", "probabilities": {"a": 0.25}})
    assert [r["example_id"] for r in store.load(path)] == ["test-0", "test-1"]
    assert store.load(path)[1]["probabilities"] == {"a": 0.25}


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert store.load(tmp_path / "nothing.jsonl") == []


def test_a_half_written_final_line_is_dropped_not_raised(tmp_path):
    path = tmp_path / "records.jsonl"
    store.append(path, {"arm": "A", "example_id": "test-0"})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"arm": "A", "exam')
    assert [r["example_id"] for r in store.load(path)] == ["test-0"]


def test_done_keys_are_what_a_resumed_run_skips(tmp_path):
    path = tmp_path / "records.jsonl"
    store.append(path, {"arm": "A", "example_id": "test-0"})
    store.append(path, {"arm": "B-512", "example_id": "test-0"})
    assert store.done_keys(path) == {("A", "test-0"), ("B-512", "test-0")}


def test_pending_keeps_only_the_work_that_is_not_already_recorded(tmp_path):
    path = tmp_path / "records.jsonl"
    store.append(path, {"arm": "A", "example_id": "test-1"})
    rows = [{"id": "test-0"}, {"id": "test-1"}, {"id": "test-2"}]
    assert [r["id"] for r in store.pending(rows, "A", store.done_keys(path))] == ["test-0", "test-2"]

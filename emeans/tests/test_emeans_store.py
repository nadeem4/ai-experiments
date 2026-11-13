"""The resumable JSONL store: appended, flushed, and safe to resume after a kill."""
import json

import pytest

from emeans import store

FRAGMENT = '{"part": "a", "dat'


def rec(arm="lloyd", seed=0, **extra):
    return {"part": "a", "dataset": "sift", "arm": arm, "setting": "iters=25", "seed": seed, **extra}


def seeds(path):
    return [r["seed"] for r in store.load(path)]


def test_append_then_load_keeps_order(tmp_path):
    path = tmp_path / "runs" / "a.jsonl"
    store.append(path, rec(seed=0))
    store.append(path, rec(seed=1))
    assert seeds(path) == [0, 1]


def test_load_of_a_missing_file_is_empty(tmp_path):
    assert store.load(tmp_path / "none.jsonl") == []


def test_a_half_written_last_line_is_dropped(tmp_path):
    path = tmp_path / "a.jsonl"
    store.append(path, rec(seed=0))
    with path.open("a", encoding="utf-8") as f:
        f.write(FRAGMENT)
    assert seeds(path) == [0]


def test_append_after_a_truncated_line_does_not_glue_records(tmp_path):
    """A resumed run must not append straight after the fragment."""
    path = tmp_path / "a.jsonl"
    store.append(path, rec(seed=0))
    with path.open("a", encoding="utf-8") as f:
        f.write(FRAGMENT)
    store.append(path, rec(seed=1))
    assert seeds(path) == [0, 1]


def test_append_repairs_a_file_with_only_a_fragment(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(FRAGMENT, encoding="utf-8")
    store.append(path, rec(seed=3))
    assert seeds(path) == [3]


def test_a_complete_last_line_without_newline_is_kept(tmp_path):
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(rec(seed=0)), encoding="utf-8")
    store.append(path, rec(seed=1))
    assert seeds(path) == [0, 1]


def test_key_and_done_keys(tmp_path):
    path = tmp_path / "a.jsonl"
    store.append(path, rec(arm="lloyd", seed=0))
    store.append(path, rec(arm="emeans", seed=1))
    assert store.key(rec()) == ("a", "sift", "lloyd", "iters=25", 0)
    assert store.done_keys(path) == {("a", "sift", "lloyd", "iters=25", 0),
                                     ("a", "sift", "emeans", "iters=25", 1)}


def test_a_record_missing_a_key_field_raises(tmp_path):
    bad = rec()
    del bad["setting"]
    with pytest.raises(KeyError):
        store.key(bad)
    with pytest.raises(KeyError):
        store.append(tmp_path / "a.jsonl", bad)

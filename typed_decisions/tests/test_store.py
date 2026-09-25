"""The append-and-flush JSONL log, which doubles as the resume point.

The key is **(spec hash, model, task, arm, example, pass)**. The spec hash is in
it because the promise of this experiment is that a model added next month can be
compared against today's numbers without re-running anything -- and that promise
is only kept if a changed spec makes yesterday's records stop counting as done.
Without the hash in the key, editing the instruction text would leave the run
"complete" and the comparison quietly invalid.
"""
from typed_decisions import store

HASH = "a" * 64
OTHER = "b" * 64


def record(**over):
    base = {"spec_hash": HASH, "model": "gpt", "task": "ag_news", "arm": "main",
            "example_id": "test-1", "repeat": 0}
    base.update(over)
    return base


class TestKey:
    def test_the_key_is_the_spec_the_model_the_task_the_arm_the_example_and_the_pass(self):
        assert store.key(record()) == (HASH, "gpt", "ag_news", "main", "test-1", 0)

    def test_a_different_spec_hash_is_different_work(self):
        """The whole point: re-deriving the example list or rewording the
        instruction must make every existing record stop matching."""
        assert store.key(record()) != store.key(record(spec_hash=OTHER))

    def test_a_different_arm_is_different_work(self):
        """`main` and `gold:last` ask the same example with the options in a
        different order. They are two calls, not one."""
        assert store.key(record()) != store.key(record(arm="gold:last"))

    def test_the_repeat_pass_is_in_the_key_so_a_second_answer_is_not_a_duplicate(self):
        assert store.key(record()) != store.key(record(repeat=1))

    def test_a_record_from_before_these_fields_existed_does_not_crash(self):
        """It reads as an unknown spec on the main arm, first pass -- and an
        unknown spec is what the report refuses to mix, which is correct."""
        assert store.key({"model": "gpt", "task": "highway", "example_id": "e"}) == \
               (None, "gpt", "highway", "main", "e", 0)


class TestRoundTrip:
    def test_append_then_load_round_trips(self, tmp_path):
        path = tmp_path / "runs" / "calls.jsonl"
        store.append(path, record(example_id="a", latency_ms=1.5))
        store.append(path, record(example_id="b", latency_ms=2.5))
        assert [r["example_id"] for r in store.load(path)] == ["a", "b"]
        assert store.load(path)[1]["latency_ms"] == 2.5

    def test_load_of_a_missing_file_is_empty(self, tmp_path):
        assert store.load(tmp_path / "nothing.jsonl") == []

    def test_a_half_written_final_line_is_dropped_not_raised(self, tmp_path):
        path = tmp_path / "calls.jsonl"
        store.append(path, record(example_id="a"))
        with path.open("a", encoding="utf-8") as f:
            f.write('{"model": "gp')
        assert [r["example_id"] for r in store.load(path)] == ["a"]

    def test_both_answers_are_kept_rather_than_the_second_overwriting(self, tmp_path):
        path = tmp_path / "calls.jsonl"
        store.append(path, record(repeat=0, choice="World"))
        store.append(path, record(repeat=1, choice="Sports"))
        assert [(r["repeat"], r["choice"]) for r in store.load(path)] == \
               [(0, "World"), (1, "Sports")]


class TestPending:
    def test_only_the_work_that_is_not_already_recorded(self, tmp_path):
        path = tmp_path / "calls.jsonl"
        store.append(path, record(example_id="b"))
        rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
        done = store.done_keys(path)
        assert [r["id"] for r in store.pending(rows, HASH, "gpt", "ag_news", "main", done)] == \
               ["a", "c"]

    def test_the_same_example_under_another_model_is_still_pending(self, tmp_path):
        """`run --models <new>` fills only what is missing for that model, and
        runs nothing else."""
        path = tmp_path / "calls.jsonl"
        store.append(path, record(example_id="a"))
        done = store.done_keys(path)
        assert [r["id"] for r in store.pending([{"id": "a"}], HASH, "phi", "ag_news", "main", done)] \
               == ["a"]

    def test_a_changed_spec_makes_finished_work_pending_again(self, tmp_path):
        """The failure this guards: the instruction text is reworded, the run is
        restarted, everything is skipped as already done, and the resulting table
        mixes two experiments."""
        path = tmp_path / "calls.jsonl"
        store.append(path, record(example_id="a"))
        done = store.done_keys(path)
        assert store.pending([{"id": "a"}], HASH, "gpt", "ag_news", "main", done) == []
        assert [r["id"] for r in store.pending([{"id": "a"}], OTHER, "gpt", "ag_news", "main", done)] \
               == ["a"]

    def test_a_bias_arm_is_pending_even_when_the_main_arm_is_done(self, tmp_path):
        path = tmp_path / "calls.jsonl"
        store.append(path, record(example_id="a", arm="main"))
        done = store.done_keys(path)
        assert [r["id"] for r in store.pending([{"id": "a"}], HASH, "gpt", "ag_news",
                                               "rand:0", done)] == ["a"]

    def test_pending_is_per_pass(self, tmp_path):
        path = tmp_path / "calls.jsonl"
        store.append(path, record(example_id="a", repeat=0))
        done = store.done_keys(path)
        assert store.pending([{"id": "a"}], HASH, "gpt", "ag_news", "main", done) == []
        assert [r["id"] for r in store.pending([{"id": "a"}], HASH, "gpt", "ag_news",
                                               "main", done, repeat=1)] == ["a"]

"""The tables the report writes beside its summary.

Everything this experiment measures is rows, so everything but the provenance is
a CSV: it diffs a row at a time in git, opens in a spreadsheet, and needs no
parser. The tests are about what each row means, because a column nobody can
explain is worse than a missing one.
"""
import csv

from rerank import tables

RECORDS = [
    {"method": "jev-score", "query_id": "q1", "doc_id": "d1", "score": 2.25, "latency_ms": 310.0},
    {"method": "jev-score", "query_id": "q1", "doc_id": "d2", "score": 2.25, "latency_ms": 290.0},
    {"method": "jev-score", "query_id": "q2", "doc_id": "d1", "score": 0.50, "latency_ms": 305.0},
    {"method": "jev-score", "query_id": "q2", "doc_id": "d2", "score": None,
     "latency_ms": 0.0, "failed": True},
    {"method": "cross-encoder", "query_id": "q1", "doc_id": "d1", "score": -1.5, "latency_ms": 38.0},
]


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestScores:
    def test_a_row_per_distinct_value_shows_how_coarse_a_scorer_is(self, tmp_path):
        """Jev answers on a two-decimal grid and the cross-encoder does not. The
        count of distinct values against the count of calls is what says so."""
        path = tables.write_scores(RECORDS, tmp_path)
        rows = _read(path)
        jev = [r for r in rows if r["method"] == "jev-score"]
        assert {r["score"] for r in jev} == {"2.25", "0.5"}
        assert [r["calls"] for r in jev if r["score"] == "2.25"] == ["2"]

    def test_a_failed_call_has_no_score_and_is_not_counted_as_one(self, tmp_path):
        rows = _read(tables.write_scores(RECORDS, tmp_path))
        assert all(r["score"] not in ("", "None") for r in rows)
        assert sum(int(r["calls"]) for r in rows if r["method"] == "jev-score") == 3


class TestLatency:
    def test_a_row_per_call_so_a_distribution_can_be_drawn(self, tmp_path):
        """p50 and p95 alone cannot be turned back into a shape."""
        rows = _read(tables.write_latency(RECORDS, tmp_path))
        assert len(rows) == 4, "the failed call has no latency to report"
        assert {r["method"] for r in rows} == {"jev-score", "cross-encoder"}
        assert sorted(float(r["latency_ms"]) for r in rows if r["method"] == "jev-score") \
            == [290.0, 305.0, 310.0]


class TestPerQuery:
    def test_a_row_per_query_carries_the_floor_and_each_methods_delta(self, tmp_path):
        floor = {"q1": 0.40, "q2": 0.00}
        deltas = {"jev-score": {"q1": 0.10, "q2": 0.00}}
        path = tables.write_per_query(floor, deltas, {"q1": 2, "q2": 0}, tmp_path)
        rows = {r["query_id"]: r for r in _read(path)}
        assert rows["q1"]["floor_ndcg@10"] == "0.4"
        assert rows["q1"]["jev-score"] == "0.1"
        assert rows["q2"]["relevant"] == "0"

    def test_a_query_with_no_relevant_candidate_is_marked_as_unable_to_move(self, tmp_path):
        """91 of 323 queries retrieved nothing relevant, so no re-ranker could
        have changed their score. They dilute every mean and the column says so."""
        path = tables.write_per_query({"q1": 0.4, "q2": 0.0}, {"jev-score": {"q1": 0.1, "q2": 0.0}},
                                      {"q1": 2, "q2": 0}, tmp_path)
        rows = {r["query_id"]: r for r in _read(path)}
        assert rows["q2"]["can_move"] == "False"
        assert rows["q1"]["can_move"] == "True"

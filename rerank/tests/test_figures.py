"""The figures must come out of the results, and must refuse rather than mislead.

Same two properties as the other experiments: the report draws them so nobody has
to remember to, and a panel whose data is missing raises instead of rendering an
empty chart that a reader takes for a measurement.
"""
import pytest

from rerank import figures

SUMMARIES = [
    {"method": "bm25", "ndcg@10": 0.3067, "latency": {"p50_ms": None, "p95_ms": None}},
    {"method": "jev-score", "ndcg@10": 0.3414,
     "ndcg@10_vs_bm25": {"mean": 0.0347, "ci95": [0.0218, 0.0476],
                         "better": 117, "worse": 52, "same": 154},
     "latency": {"p50_ms": 331.0, "p95_ms": 677.0},
     "scores": {"distinct": 389, "of": 6147, "tied": 1748}},
    {"method": "cross-encoder", "ndcg@10": 0.3272,
     "ndcg@10_vs_bm25": {"mean": 0.0204, "ci95": [0.0092, 0.0317],
                         "better": 97, "worse": 67, "same": 159},
     "latency": {"p50_ms": 37.5, "p95_ms": 76.2},
     "scores": {"distinct": 6389, "of": 6460, "tied": 136}},
]
DELTAS = {"jev-score": {"q1": 0.1, "q2": 0.0, "q3": -0.05},
          "cross-encoder": {"q1": 0.05, "q2": 0.0, "q3": 0.02}}
RELEVANT = {"q1": 2, "q2": 0, "q3": 1}
RECORDS = [
    {"method": "jev-score", "query_id": "q1", "doc_id": "d1", "score": 2.25, "latency_ms": 310.0},
    {"method": "cross-encoder", "query_id": "q1", "doc_id": "d1", "score": -1.5, "latency_ms": 38.0},
]


def results(**over):
    return {"dataset": "BEIR NFCorpus (test)", "tag": "test", "methods": SUMMARIES, **over}


class TestWriteAll:
    def test_a_complete_result_draws_every_figure(self, tmp_path):
        out = figures.write_all(results(), DELTAS, RELEVANT, RECORDS, tmp_path)
        assert out["skipped"] == {}
        assert sorted(out["written"]) == ["latency.png", "per-query-delta.png",
                                          "score-distribution.png", "win-loss.png"]
        for name in out["written"]:
            assert (tmp_path / name).stat().st_size > 0

    def test_a_panel_with_no_data_is_skipped_with_its_reason(self, tmp_path):
        out = figures.write_all(results(), {}, RELEVANT, [], tmp_path)
        assert "per-query-delta" in out["skipped"]
        assert not (tmp_path / "per-query-delta.png").exists(), "a skip writes no file"


class TestRefusals:
    def test_the_delta_panel_needs_per_query_numbers(self, tmp_path):
        with pytest.raises(figures.NothingToPlot):
            figures.per_query_delta_plot(results(), {}, RELEVANT, tmp_path)

    def test_the_score_panel_needs_scores(self, tmp_path):
        with pytest.raises(figures.NothingToPlot):
            figures.score_distribution_plot(results(), [], tmp_path)

    def test_the_win_loss_panel_needs_a_method_measured_against_bm25(self, tmp_path):
        only_floor = results(methods=[SUMMARIES[0]])
        with pytest.raises(figures.NothingToPlot):
            figures.win_loss_plot(only_floor, tmp_path)


class TestTheDeltaPanelSeparatesQueriesThatCannotMove:
    """91 of the 323 queries retrieved nothing relevant. Drawing them among the
    rest makes a wall of exact zeros look like a model declining to act."""

    def test_it_counts_them_rather_than_hiding_them(self):
        movable, stuck = figures.split_by_movability(DELTAS["jev-score"], RELEVANT)
        assert sorted(movable) == ["q1", "q3"]
        assert stuck == ["q2"]

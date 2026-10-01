"""What the site is given, and what it deliberately is not.

The page lets a reader watch any query re-rank. That is only possible if every
query ships, which is only affordable if the wire log does not: 25,840 records of
roughly 2KB cannot go to a browser, but the order each method produced can.
"""
import pytest

from rerank.scripts import site_data

CANDIDATES = {"q1": ["d1", "d2", "d3"], "q2": ["d4", "d5"]}
RECORDS = [
    {"method": "jev-score", "query_id": "q1", "doc_id": "d1", "score": 1.0},
    {"method": "jev-score", "query_id": "q1", "doc_id": "d2", "score": 3.0},
    {"method": "jev-score", "query_id": "q1", "doc_id": "d3", "score": 2.0},
]
CORPUS = {d: {"title": f"Title {d}", "text": f"Body of {d}. " * 40}
          for d in ("d1", "d2", "d3", "d4", "d5")}
QRELS = {"q1": {"d2": 2, "d9": 1}, "q2": {}}
QUERIES = {"q1": "a question", "q2": "another question"}


class TestTheOrderEachMethodProduced:
    def test_a_query_carries_every_methods_ranking(self):
        out = site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {})
        q1 = next(q for q in out["queries"] if q["id"] == "q1")
        assert q1["orders"]["hybrid"] == ["d1", "d2", "d3"], "the floor is the candidate order"
        assert q1["orders"]["jev-score"] == ["d2", "d3", "d1"], "sorted by score, best first"

    def test_a_method_that_did_not_score_a_query_is_absent_not_guessed(self):
        out = site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {})
        q2 = next(q for q in out["queries"] if q["id"] == "q2")
        assert "jev-score" not in q2["orders"]
        assert q2["orders"]["hybrid"] == ["d4", "d5"]

    def test_relevance_is_carried_only_for_the_candidates(self):
        """The judgements hold documents the retriever never returned. Shipping
        them would imply the re-ranker could have reached them."""
        out = site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {})
        q1 = next(q for q in out["queries"] if q["id"] == "q1")
        assert q1["relevant"] == ["d2"], "d9 is judged relevant but was never a candidate"

    def test_a_query_that_could_not_move_is_marked(self):
        out = site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {})
        assert next(q for q in out["queries"] if q["id"] == "q1")["can_move"] is True
        assert next(q for q in out["queries"] if q["id"] == "q2")["can_move"] is False


class TestTheCorpusItShips:
    def test_only_documents_some_query_actually_retrieved(self):
        out = site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {})
        assert set(out["docs"]) == {"d1", "d2", "d3", "d4", "d5"}

    def test_passages_are_cut_so_the_payload_stays_small(self):
        out = site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {})
        assert all(len(d["text"]) <= site_data.SNIPPET for d in out["docs"].values())
        assert out["docs"]["d1"]["title"] == "Title d1"

    def test_the_whole_payload_is_small_enough_for_a_browser(self):
        """The wire log is 50MB. This is the derived order, which is not."""
        import json
        out = site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {})
        assert len(json.dumps(out)) < 200_000


class TestScoresAreCarriedForTheTheatre:
    def test_each_ranking_keeps_the_number_that_produced_it(self):
        out = site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {})
        q1 = next(q for q in out["queries"] if q["id"] == "q1")
        assert q1["scores"]["jev-score"]["d2"] == 3.0

    def test_a_failed_call_keeps_its_position_and_the_query_still_ranks(self):
        """The run ranked around its one failed call rather than dropping the
        query, and reported a delta for it. The page must show that same order."""
        records = RECORDS + [
            {"method": "jev-score", "query_id": "q2", "doc_id": "d4",
             "score": None, "failed": True},
            {"method": "jev-score", "query_id": "q2", "doc_id": "d5", "score": 5.0},
        ]
        out = site_data.build(CANDIDATES, records, CORPUS, QRELS, QUERIES, {})
        q2 = next(q for q in out["queries"] if q["id"] == "q2")
        assert q2["orders"]["jev-score"] == ["d4", "d5"], "d4 holds the slot BM25 gave it"
        assert q2["scores"]["jev-score"]["d4"] is None, "no number was returned for it"

    def test_a_query_the_method_never_finished_is_absent_not_guessed(self):
        """Different from a failed call: d5 was never attempted, so there is no
        order to show. Ranking on a hole would invent one."""
        records = RECORDS + [{"method": "jev-score", "query_id": "q2", "doc_id": "d4",
                              "score": 1.0}]
        out = site_data.build(CANDIDATES, records, CORPUS, QRELS, QUERIES, {})
        assert "jev-score" not in next(q for q in out["queries"] if q["id"] == "q2")["orders"]


class TestTheSplit:
    """The index paints the page; the detail is fetched when a query is opened."""

    def _split(self):
        return site_data.split(site_data.build(CANDIDATES, RECORDS, CORPUS, QRELS, QUERIES, {}))

    def test_the_index_carries_titles_but_not_passages(self):
        index, _ = self._split()
        assert index["titles"]["d1"] == "Title d1"
        assert "passages" not in index

    def test_the_index_carries_no_orders_or_scores(self):
        index, _ = self._split()
        assert all("orders" not in q and "scores" not in q for q in index["queries"])

    def test_scores_travel_aligned_to_the_order_not_keyed_by_id(self):
        """Repeating the document ids was a third of the payload."""
        _, detail = self._split()
        q1 = detail["queries"]["q1"]
        assert q1["orders"]["jev-score"] == ["d2", "d3", "d1"]
        assert q1["scores"]["jev-score"] == [3.0, 2.0, 1.0], "same order as the ranking"

    def test_the_detail_carries_the_passages(self):
        _, detail = self._split()
        assert detail["passages"]["d1"].startswith("Body of d1.")

    def test_a_failed_calls_slot_in_the_score_array_is_null(self):
        """The arrays stay aligned to the order, so the hole has to be held."""
        records = RECORDS + [
            {"method": "jev-score", "query_id": "q2", "doc_id": "d4",
             "score": None, "failed": True},
            {"method": "jev-score", "query_id": "q2", "doc_id": "d5", "score": 5.0},
        ]
        _, detail = site_data.split(
            site_data.build(CANDIDATES, records, CORPUS, QRELS, QUERIES, {}))
        assert detail["queries"]["q2"]["scores"]["jev-score"] == [None, 5.0]

"""Parsing the official BEIR qrels file. Nothing here touches the network."""
import pytest

from rerank.data import parse_qrels

TSV = """query-id\tcorpus-id\tscore
PLAIN-2\tMED-2427\t2
PLAIN-2\tMED-2428\t1
PLAIN-3\tMED-10\t0
"""


def test_grades_are_kept_per_query():
    assert parse_qrels(TSV) == {"PLAIN-2": {"MED-2427": 2, "MED-2428": 1}, "PLAIN-3": {"MED-10": 0}}


def test_the_header_is_not_a_judgement():
    assert "query-id" not in parse_qrels(TSV)


def test_blank_lines_are_skipped():
    assert parse_qrels(TSV + "\n\n") == parse_qrels(TSV)


def test_a_malformed_row_is_an_error():
    """A silently dropped judgement would quietly change every number."""
    with pytest.raises(ValueError):
        parse_qrels("query-id\tcorpus-id\tscore\nPLAIN-2\tMED-2427\n")

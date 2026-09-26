"""Which queries the site gets to show.

The full wire is 25,840 records of roughly 3 KB, so the browsable examples are a
curated subset while the table stays over all 323 queries. The selection is the
one piece of that with a decision in it, so it is a pure function and it is
tested: it has to span the outcomes rather than pick the prettiest ones.
"""
from rerank.examples import QUOTAS, select_examples


def q(qid, jev=0.0, laya=0.0, failed=0):
    return {"query_id": qid, "jev_delta": jev, "laya_delta": laya, "jev_failed": failed}


def many(n, **kw):
    return [q(f"Q{i:03d}", **kw) for i in range(n)]


def reasons(picked):
    return {p["query_id"]: p["reason"] for p in picked}


def test_nothing_to_choose_from_picks_nothing():
    assert select_examples([]) == []


def test_the_biggest_jev_gains_are_picked_over_the_smaller_ones():
    queries = [q("small", jev=0.01), q("big", jev=0.9), q("mid", jev=0.4)]
    picked = [p["query_id"] for p in select_examples(queries, total=2)]
    assert picked[:2] == ["big", "mid"]


def test_the_biggest_laya_losses_are_picked_over_the_smaller_ones():
    queries = [q("small", laya=-0.01), q("worst", laya=-0.9), q("mid", laya=-0.4)]
    picked = reasons(select_examples(queries))
    assert picked["worst"] == "laya-loss" and picked["mid"] == "laya-loss"


def test_queries_where_nothing_moved_are_included_too():
    """A re-ranker that agrees with BM25 is a result, and the page says so."""
    picked = reasons(select_examples([q("flat"), q("gain", jev=0.5)]))
    assert picked["flat"] == "unchanged"


def test_a_query_with_a_failed_jev_call_is_always_in():
    """4.85% of Jev's calls never succeeded and those passages kept their BM25
    slot. That has to be visible on the page, so it is picked first."""
    queries = many(40, jev=0.5) + [q("broke", jev=0.001, failed=3)]
    picked = reasons(select_examples(queries))
    assert picked["broke"] == "jev-failure"


def test_the_worst_failure_is_the_one_shown():
    queries = [q("one", failed=1), q("seven", failed=7), q("two", failed=2)]
    picked = [p["query_id"] for p in select_examples(queries, total=1)]
    assert picked == ["seven"]


def test_a_query_is_picked_once_and_carries_one_reason():
    queries = many(80, jev=0.3, laya=-0.3, failed=1)
    picked = select_examples(queries)
    ids = [p["query_id"] for p in picked]
    assert len(ids) == len(set(ids))
    assert all(p["reason"] in QUOTAS for p in picked)


def test_it_asks_for_about_two_dozen_and_spans_every_outcome():
    queries = (many(30, jev=0.3) + [q(f"L{i}", laya=-0.3) for i in range(30)]
               + [q(f"F{i}") for i in range(30)] + [q("broke", failed=4), q("broke2", failed=1)])
    picked = select_examples(queries)
    assert len(picked) == sum(QUOTAS.values()) == 24
    assert set(p["reason"] for p in picked) == set(QUOTAS)


def test_a_bucket_with_nothing_in_it_is_left_short_rather_than_padded():
    """Better a smaller honest subset than one topped up with queries that do not
    show what the bucket claims to show."""
    picked = select_examples(many(40, jev=0.3))
    assert [p["reason"] for p in picked] == ["jev-gain"] * QUOTAS["jev-gain"]


def test_the_same_input_always_gives_the_same_subset():
    queries = many(50, jev=0.3, laya=-0.3)
    assert select_examples(queries) == select_examples(list(reversed(queries)))

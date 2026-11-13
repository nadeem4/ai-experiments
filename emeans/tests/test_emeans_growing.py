"""The growing index on a short synthetic timeline: 2019-12 is the initial
period, then five arrival months of 40 docs with 5 held out as queries."""
import numpy as np
import pytest

from emeans import growing

MONTHS = [f"2020-{m:02d}" for m in range(1, 6)]
N_Q = 5


def month_ids():
    return np.array(["2019-12"] * 30 + [m for m in MONTHS for _ in range(40)])


def test_timeline_is_2020_01_onwards_and_72_at_full_scale():
    t = growing.timeline(72)
    assert len(t) == 72 and t[0] == "2020-01" and t[-1] == "2025-12"
    assert growing.timeline(3) == ["2020-01", "2020-02", "2020-03"]
    assert growing.INITIAL == ("2018-01", "2019-12")


def test_held_out_queries_are_fixed_and_do_not_depend_on_the_run_seed():
    a = growing.hold_out(month_ids(), n=N_Q, dataset_seed=0)
    b = growing.hold_out(month_ids(), n=N_Q, dataset_seed=0)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert len(a[0]) == N_Q * len(MONTHS)
    assert not np.array_equal(a[0], growing.hold_out(month_ids(), n=N_Q, dataset_seed=1)[0])


def test_queries_and_indexed_are_disjoint_and_cover_everything():
    queries, indexed = growing.hold_out(month_ids(), n=N_Q)
    assert not set(queries) & set(indexed)
    assert sorted(set(queries) | set(indexed)) == list(range(len(month_ids())))


def test_a_query_is_in_no_indexed_prefix_initial_set_or_training_set():
    """Review Focus 2: leakage pin."""
    ids = month_ids()
    queries, indexed = growing.hold_out(ids, n=N_Q)
    indexed_months = ids[indexed]
    initial = indexed[:growing.n_through(indexed_months, "2019-12")]
    assert not set(queries) & set(initial)
    for month in MONTHS:
        rows = indexed[:growing.n_through(indexed_months, month)]
        assert not set(queries) & set(rows)
    qs = growing.make_split(np.arange(len(ids), dtype=np.float32)[:, None], ids, 5, n_queries=N_Q)
    seen = {int(v) for v in qs.vectors[:, 0]}
    held = {int(v) for q in qs.queries.values() for v in q[:, 0]}
    assert not seen & held and len(held) == N_Q * 5


def test_indexed_is_chronological_and_prefixes_grow_by_one_months_non_queries():
    ids = month_ids()
    _, indexed = growing.hold_out(ids, n=N_Q)
    months = ids[indexed]
    assert list(months) == sorted(months)
    sizes = [growing.n_through(months, m) for m in ["2019-12"] + MONTHS]
    assert sizes[0] == 30 and np.diff(sizes).tolist() == [40 - N_Q] * 5


def test_a_month_with_too_few_docs_is_named():
    ids = np.array(["2019-12"] * 10 + ["2020-01"] * 3 + ["2020-02"] * 40)
    with pytest.raises(ValueError, match="2020-01"):
        growing.hold_out(ids, n=N_Q)

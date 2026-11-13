"""Sampling the Cornell arXiv metadata: cs-primary papers, one sample per publish month."""
import json

from emeans import arxiv

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def rec(i, categories="cs.LG", year=2020, month=1, abstract=None):
    """A record in the real Cornell format: RFC 2822 dates, space-separated categories."""
    return {"id": f"{year}.{i:05d}", "categories": categories,
            "abstract": abstract or f" Abstract {i}.\n",
            "versions": [{"version": "v1",
                          "created": f"Mon, 2 {MONTHS[month - 1]} {year} 19:18:42 GMT"},
                         {"version": "v2", "created": "Fri, 9 Dec 2033 10:00:00 GMT"}]}


def test_primary_category_decides_not_cross_lists():
    assert not arxiv.is_cs_primary(rec(1, "math.CO cs.DM"))
    assert arxiv.is_cs_primary(rec(2, "cs.DM math.CO"))
    assert not arxiv.is_cs_primary(rec(3, "csx.AB"))


def test_month_is_the_first_versions_publish_month():
    assert arxiv.month_of(rec(1, year=2018, month=4)) == "2018-04"


def synthetic():
    records = [rec(i, month=1) for i in range(50)]            # 50 in 2020-01
    records += [rec(100 + i, month=2) for i in range(3)]      # 3 in 2020-02
    records += [rec(200, "math.CO cs.DM"), rec(201, "cs.DM math.CO")]
    records += [rec(300, year=2017, month=12), rec(301, year=2026, month=1)]
    return records


def test_range_and_cross_lists_are_excluded():
    out = arxiv.sample(synthetic(), per_month=1000, start="2018-01", end="2025-12")
    ids = {r["id"] for r in out}
    assert "2020.00200" not in ids and "2020.00201" in ids
    assert "2017.00300" not in ids and "2026.00301" not in ids


def test_cap_and_small_months():
    counts = arxiv.month_counts(arxiv.sample(synthetic(), per_month=10))
    assert counts == {"2020-01": 10, "2020-02": 3}


def test_deterministic_per_seed_and_sorted_by_month():
    ids = lambda **kw: [r["id"] for r in arxiv.sample(synthetic(), per_month=10, **kw)]
    assert ids(seed=0) == ids(seed=0)
    assert ids(seed=0) != ids(seed=1)
    months = [r["month"] for r in arxiv.sample(synthetic(), per_month=10)]
    assert months == sorted(months)


def test_sample_rows_hold_id_month_and_stripped_abstract():
    row = arxiv.sample([rec(1)], per_month=5)[0]
    assert row == {"id": "2020.00001", "month": "2020-01", "abstract": "Abstract 1."}


def test_stream_reads_a_jsonl_line_by_line(tmp_path):
    path = tmp_path / "meta.json"
    path.write_text("\n".join(json.dumps(r) for r in synthetic()) + "\n", encoding="utf-8")
    streamed = arxiv.sample(arxiv.read_jsonl(path), per_month=10)
    assert arxiv.month_counts(streamed) == {"2020-01": 10, "2020-02": 3}

"""summary.json: hand-computed values, determinism, and CSVs that add no numbers."""
import csv
import json

import pytest

from emeans import tables


@pytest.fixture
def summary(fake_records):
    return tables.summary(fake_records, n_boot=200)


def test_part_a_match_table_has_one_match_and_one_non_match(summary):
    m = summary["a"]["sift"]["match"]
    assert m["emeans"]["epochs_to_match"] == 3
    assert m["emeans"]["speedup"] == pytest.approx((100 / 30 + 200 / 40) / 2)
    assert m["emeans"]["ratio"]["mean"] == pytest.approx(0.99) and m["emeans"]["ratio"]["lo"] is not None
    assert m["minibatch"]["epochs_to_match"] is None and m["minibatch"]["speedup"] is None
    assert m["minibatch"]["ratio"]["mean"] == pytest.approx(1.1)  # reported at the largest epoch count


def test_race_curve_and_lloyds_own_curve_are_ratios_to_lloyd_25(summary):
    a = summary["a"]["sift"]
    assert a["race"]["emeans"]["1"]["mean"] == pytest.approx(1.2)
    assert a["lloyd_curve"]["25"] == pytest.approx(1.1)  # (10 + 25/25) / 10
    assert a["lloyd_curve"]["1"] == pytest.approx(3.5)


def test_budget_table_reports_actual_time_and_flags_the_overshoot(summary):
    rows = {(r["arm"], r["budget_s"]): r for r in summary["a"]["sift"]["budget"]}
    assert rows[("emeans", 10)]["overshoot"] is True and rows[("emeans", 10)]["actual_s"] == pytest.approx(12.0)
    assert rows[("emeans", 30)]["overshoot"] is False and rows[("lloyd", 10)]["overshoot"] is False
    search = {(r["arm"], r["setting"]): r for r in summary["a"]["sift"]["search"]}
    assert search[("lloyd", "iters=25")]["search"]["16"]["scanned_p95"] == 150.0


def test_part_b_has_series_ledger_and_every_pair_at_both_nprobes(summary):
    b = summary["b"]
    s = b["series"]["emeans"]
    assert s["months"][:2] == ["2020-01", "2020-02"] and len(s["months"]) == 14
    assert s["recall16"]["min"][0] <= s["recall16"]["mean"][0] <= s["recall16"]["max"][0]
    assert s["cumulative_s"][1] == pytest.approx(4.0)  # two months at (1.5 + 2.5) / 2
    assert b["ledger"]["stale"]["total_s"] == 0 and b["ledger"]["emeans"]["total_s"] == pytest.approx(14 * 2.0)
    pairs = {(e["a"], e["b"], e["nprobe"]): e for e in b["end_state"]}
    assert len(pairs) == 12
    gap = pairs[("stale", "emeans", 16)]
    assert gap["mean"] == pytest.approx(-0.04) and gap["hi"] < 0
    assert set(gap["per_seed"]) == {"0", "1"}


def test_part_c_is_by_dimension(summary):
    c = summary["c"]
    assert set(c) == {"64", "128"} and c["64"]["epochs_to_match"] == 3


def test_a_run_without_2025_has_no_end_state_but_still_a_summary(fake_records):
    recs = [r for r in fake_records if not r["setting"].startswith("2025")]
    assert tables.summary(recs, n_boot=50)["b"]["end_state"] == []


def test_summary_json_is_byte_identical_on_rerun(fake_records, tmp_path):
    a = tables.write_summary(tables.summary(fake_records, n_boot=200), tmp_path / "a")
    b = tables.write_summary(tables.summary(list(reversed(fake_records)), n_boot=200), tmp_path / "b")
    assert a.read_bytes() == b.read_bytes()
    assert json.loads(a.read_text())["a"]["sift"]["match"]["emeans"]["epochs_to_match"] == 3


def numbers(obj):
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():  # a dimension is a key ("128"), still a summary.json number
            if k.isdigit():
                yield float(k)
            yield from numbers(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from numbers(v)


def test_every_csv_number_comes_from_summary_json(summary, tmp_path):
    paths = tables.write_csv(summary, tmp_path)
    assert {p.name for p in paths} >= {"a_match.csv", "a_budget.csv", "b_end_state.csv", "b_ledger.csv", "c_dimension.csv"}
    known = set(numbers(summary))
    for p in paths:
        for row in csv.DictReader(p.open(encoding="utf-8")):
            for cell in row.values():
                try:
                    v = float(cell)
                except ValueError:
                    continue
                assert any(v == pytest.approx(k) for k in known), f"{p.name}: {cell} is not in summary.json"


def test_cumulative_maintenance_carries_the_per_seed_range(summary):
    r = summary["b"]["series"]["emeans"]["cumulative_range"]
    # seeds spend 1.5 s and 2.5 s a month, so after two months 3.0 and 5.0
    assert (r["min"][1], r["max"][1]) == pytest.approx((3.0, 5.0))
    assert r["min"][-1] <= summary["b"]["series"]["emeans"]["cumulative_s"][-1] <= r["max"][-1]

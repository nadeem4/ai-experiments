"""Figures are drawn from summary.json alone, in greyscale, and skip honestly."""
import json

import pytest

from emeans import figures, tables

NAMES = {"b-recall-over-time", "b-cumulative-maintenance", "b-end-state-gap", "a-race", "a-budget-recall",
         "a-scanned-p95", "c-dimension"}


@pytest.fixture
def summary(fake_records):
    # a JSON round trip: what the figures see is exactly what is on disk
    return json.loads(json.dumps(tables.summary(fake_records, n_boot=100)))


def test_every_figure_is_a_png_for_a_full_record_set(summary, tmp_path):
    out = figures.write_all(summary, tmp_path)
    assert set(out) == NAMES
    for name, value in out.items():
        assert (tmp_path / f"{name}.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", value
        assert value == str(tmp_path / f"{name}.png")


def test_a_part_with_no_records_is_skipped_with_a_reason_and_no_file(summary, tmp_path):
    only_a = {"a": summary["a"]}
    out = figures.write_all(only_a, tmp_path)
    assert out["a-race"] == str(tmp_path / "a-race.png")
    for name in ("b-recall-over-time", "b-cumulative-maintenance", "b-end-state-gap", "c-dimension"):
        assert out[name].startswith("skipped: ") and not (tmp_path / f"{name}.png").exists()


def test_without_2025_the_end_state_figure_is_skipped_not_drawn_empty(summary, tmp_path):
    summary["b"]["end_state"] = []
    out = figures.write_all(summary, tmp_path)
    assert out["b-end-state-gap"].startswith("skipped: ") and "2025" in out["b-end-state-gap"]


def test_a_part_a_without_budgets_skips_the_budget_figures(fake_records, tmp_path):
    no_budget = [r for r in fake_records if not r["setting"].startswith("budget=")]
    out = figures.write_all(tables.summary(no_budget, n_boot=50), tmp_path)
    assert out["a-budget-recall"].startswith("skipped: ") and out["a-scanned-p95"].startswith("skipped: ")
    assert out["a-race"].endswith("a-race.png")


def test_the_styling_is_greyscale():
    for c in figures.SHADES:
        assert len(c) == 7 and c[1:3] == c[3:5] == c[5:7]

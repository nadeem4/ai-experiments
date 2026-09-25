"""The site's banking77 numbers are exported from the results file, never typed.

A number written into a web page by hand is right once. These tests pin the two
things that keeps honest: every headline figure is arithmetic over `results/`,
and a negative result stays negative on the way out.
"""
import json

from banking77.scripts import export_site_data


def arm(name, accuracy, **kw):
    base = {
        "arm": name, "n_test": 3080, "n_options": 77, "head_max_len": 256,
        "max_len": 1024, "checkpoint": "multilingual",
        "accuracy": accuracy, "accuracy_ci95": [accuracy - 0.01, accuracy + 0.01],
        "top5_accuracy": 0.6, "macro_f1": 0.4, "ece_raw": 0.3,
        "latency_ms_p50": 380.0, "latency_ms_p95": 450.0, "latency_device": "cpu",
        "temperature": 2.0, "n_val": 1003, "ece_after_temperature": 0.12,
        "accuracy_after_temperature": accuracy,
    }
    base.update(kw)
    return base


def results(**over):
    base = {
        "tag": "full",
        "published_comparison": {
            "source": "convaiinnovations/laya model card",
            "laya_banking77_accuracy": 0.425,
            "jev_banking77_accuracy": 0.870,
            "note": "published, not measured here",
        },
        "arms": {
            "A-default": arm("A-default", 0.40),
            "B-384": arm("B-384", 0.50, head_max_len=384),
            "B-512": arm("B-512", 0.50, head_max_len=512),
            "B-768": arm("B-768", 0.50, head_max_len=768),
            "C-coarse-fine": arm("C-coarse-fine", 0.30),
        },
        "truncation_diagnostics": {
            "A-default": {
                "n_options": 77, "mean_text_tokens_per_option": 2.74,
                "truncated_fraction": 0.35, "truncated_options": 27,
                "distinct_options": 73, "identical_pairs": 5, "collided_options": 7,
                "head_max_len": 256, "max_len": 1024, "sequence_tokens": 303,
                "collisions": {" lost or stolen": ["lost or stolen card",
                                                   "lost or stolen phone"]},
            },
            "C-coarse-fine": {"coarse": {"n_options": 11}, "fine_worst_group": {"n_options": 7}},
        },
        "option_count_sweep": [
            {"arm": "K-10", "n_options": 10, "tokens_per_option": 3.0,
             "identical_pairs": 0, "accuracy": 0.74, "accuracy_ci95": [0.70, 0.79]},
        ],
        "mcnemar": [
            {"a": "A-default", "b": "B-384", "n_paired": 3080, "accuracy_a": 0.40,
             "accuracy_b": 0.50, "n10": 151, "n01": 299, "n_discordant": 450,
             "p_value": 2.7e-12},
        ],
    }
    base.update(over)
    return base


def test_the_headline_is_arithmetic_over_the_arms():
    head = export_site_data.build(results())["headline"]
    # 0.870 published, 0.40 measured at the shipped budget, 0.50 with room.
    assert round(head["deficit_points"], 2) == 47.00
    assert round(head["budget_gain_points"], 2) == 10.00
    assert round(head["remaining_points"], 2) == 37.00
    assert round(head["share_of_deficit"], 4) == round(10.0 / 47.0, 4)


def test_a_workaround_that_lost_exports_as_a_loss():
    head = export_site_data.build(results())["headline"]
    assert head["workaround_vs_default_points"] < 0
    assert head["workaround_vs_raised_points"] < 0
    assert round(head["workaround_vs_default_points"], 2) == -10.00
    assert round(head["workaround_vs_raised_points"], 2) == -20.00


def test_arms_that_agree_to_every_decimal_are_named_as_a_cliff():
    head = export_site_data.build(results())["headline"]
    assert head["identical_budget_arms"] == ["B-384", "B-512", "B-768"]
    assert head["saturates_at"] == 384


def test_nested_diagnostics_stay_out_of_the_flat_truncation_table():
    # Arm C's diagnostics are one block per step and have no top-level option
    # count, so a table that assumed one row per arm would print nothing for it.
    arms = [row["arm"] for row in export_site_data.build(results())["truncation"]]
    assert arms == ["A-default"]


def test_the_collisions_the_page_prints_come_from_the_default_arm():
    built = export_site_data.build(results())
    assert built["collisions"] == {" lost or stolen": ["lost or stolen card",
                                                       "lost or stolen phone"]}


def test_an_arm_carries_only_what_the_page_renders():
    row = export_site_data.build(results())["arms"][0]
    assert set(row) == {"arm", "n_test", "n_options", "head_max_len", "checkpoint",
                        "accuracy", "ci95", "top5", "macro_f1", "ece_raw",
                        "temperature", "ece_after_temperature", "p50_ms", "p95_ms"}


def test_the_committed_results_file_still_says_what_the_page_says():
    """The one test that fails if the sweep is re-run and the page goes stale."""
    built = export_site_data.build(json.loads(export_site_data.RESULTS.read_text("utf-8")))
    head = built["headline"]
    assert round(head["budget_gain_points"], 2) == 4.81
    assert round(head["deficit_points"], 2) == 42.97
    assert round(head["remaining_points"], 2) == 38.17
    assert round(head["share_of_deficit"], 3) == 0.112
    assert round(head["workaround_vs_default_points"], 2) == -9.61
    assert round(head["workaround_vs_raised_points"], 2) == -14.42
    assert head["saturates_at"] == 384
    assert len(built["collisions"]) == 3


def test_write_lands_where_the_page_imports_it(tmp_path):
    out = tmp_path / "banking77.json"
    export_site_data.write(results(), out, source=tmp_path, public=tmp_path / "public")
    assert json.loads(out.read_text("utf-8"))["headline"]["saturates_at"] == 384


# 1x1 PNG: enough for the IHDR the exporter reads its dimensions from.
PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000050001"
    "0d0a2db40000000049454e44ae426082")


def test_figures_are_copied_with_the_size_the_page_reserves(tmp_path):
    source = tmp_path / "figures"
    source.mkdir()
    for name in export_site_data.SHOWN_FIGURES.values():
        (source / name).write_bytes(PNG)
    out = tmp_path / "public"
    figures = export_site_data.copy_figures(source, out)
    assert figures["arms"]["src"] == "/banking77-arms.png"
    assert figures["arms"]["width"] == 1 and figures["arms"]["height"] == 1
    assert (out / "banking77-option-count.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_figure_the_report_did_not_draw_is_absent_rather_than_broken(tmp_path):
    source = tmp_path / "figures"
    source.mkdir()
    (source / "arms.png").write_bytes(PNG)
    figures = export_site_data.copy_figures(source, tmp_path / "public")
    assert set(figures) == {"arms"}

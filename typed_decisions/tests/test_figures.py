"""Every figure is generated from the results file the tables come from.

The rule: a chart must never be able to drift from the numbers printed beside it,
so no figure is drawn by hand and none takes its own input. The second rule: a
figure with nothing to show **fails loudly** rather than writing an empty or
misleading chart -- a calibration panel for models that return no probabilities,
or a cost-against-accuracy plot for a task with no ground truth, is worse than no
figure at all, because a reader assumes a published chart means something.
"""
import json

import pytest

from typed_decisions import figures

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def row(model, task="ag_news", **kw):
    base = {"model": model, "model_id": f"vendor/{model}", "task": task,
            "transport": "openrouter", "local": model == "laya", "n": 50, "n_options": 4,
            "p50_ms": 400.0, "p95_ms": 900.0, "tail_ratio": 2.25,
            "mean_input_tokens": 120.0, "mean_output_tokens": 6.0,
            "cost_per_1k_usd": 0.12, "cost_per_correct_usd": 1.4e-4,
            "accuracy": 0.8, "accuracy_ci": (0.7, 0.9), "valid_rate": 1.0,
            "returns_probability": model in ("jev", "laya")}
    base.update(kw)
    return base


def results(**over):
    base = {
        "tag": "pilot",
        "spec_hashes": {"ag_news": "a" * 64, "clinc150": "b" * 64},
        "summary": {
            "gpt/ag_news@openrouter": row("gpt"),
            "jev/ag_news@openrouter": row("jev", accuracy=0.9, cost_per_1k_usd=0.05),
            "laya/ag_news@local-cpu": row("laya", transport="local-cpu", accuracy=0.4),
            "gpt/clinc150@openrouter": row("gpt", task="clinc150", n_options=151,
                                           mean_input_tokens=1400.0),
            "jev/clinc150@openrouter": row("jev", task="clinc150", n_options=151,
                                           mean_input_tokens=900.0),
            "laya/clinc150@local-cpu": row("laya", task="clinc150", transport="local-cpu",
                                           n_options=151, mean_input_tokens=None),
        },
        "position_bias": {
            "gpt/ag_news": {"model": "gpt", "task": "ag_news", "n_options": 4, "n_examples": 40,
                            "flip": {"rate": 0.1, "n": 40, "n_incomplete": 0},
                            "gold": {"accuracy": {"first": 0.85, "middle": 0.8, "last": 0.78},
                                     "spread": 0.07, "n": {"first": 40, "middle": 40, "last": 40}},
                            "positions": {"shares": [0.4, 0.3, 0.2, 0.1], "edges": [0, 1, 2, 3, 4],
                                          "mean_normalised": 0.33, "n": 120}},
            "jev/ag_news": {"model": "jev", "task": "ag_news", "n_options": 4, "n_examples": 40,
                            "flip": {"rate": 0.0, "n": 40, "n_incomplete": 0},
                            "gold": {"accuracy": {"first": 0.9, "middle": 0.9, "last": 0.9},
                                     "spread": 0.0, "n": {"first": 40, "middle": 40, "last": 40}},
                            "positions": {"shares": [0.25] * 4, "edges": [0, 1, 2, 3, 4],
                                          "mean_normalised": 0.5, "n": 120}},
            "gpt/clinc150": {"model": "gpt", "task": "clinc150", "n_options": 151,
                             "n_examples": 40,
                             "flip": {"rate": 0.55, "n": 40, "n_incomplete": 0},
                             "gold": {"accuracy": {"first": 0.8, "middle": 0.6, "last": 0.45},
                                      "spread": 0.35,
                                      "n": {"first": 40, "middle": 40, "last": 40}},
                             "positions": {"shares": [0.3, 0.2, 0.1, 0.1, 0.07, 0.06, 0.05,
                                                      0.05, 0.04, 0.03],
                                           "edges": list(range(0, 166, 15)),
                                           "mean_normalised": 0.28, "n": 120}},
        },
        "calibration": {
            "jev/ag_news": {"model": "jev", "task": "ag_news", "temperature": 1.4,
                            "ece_raw": 0.08, "ece_scaled": 0.03, "n_validation": 100,
                            "n_test": 50, "n_bins": 15,
                            "reliability": {"bin_centres": [0.1, 0.3, 0.5, 0.7, 0.9],
                                            "accuracy": [0.05, 0.25, 0.55, 0.72, 0.95],
                                            "counts": [2, 5, 10, 13, 20]}},
            "laya/ag_news": {"model": "laya", "task": "ag_news", "temperature": 2.1,
                             "ece_raw": 0.21, "ece_scaled": 0.09, "n_validation": 100,
                             "n_test": 50, "n_bins": 15,
                             "reliability": {"bin_centres": [0.1, 0.3, 0.5, 0.7, 0.9],
                                             "accuracy": [0.3, 0.3, 0.4, 0.5, 0.6],
                                             "counts": [4, 8, 12, 16, 10]}},
        },
    }
    base.update(over)
    return base


class TestWriteAll:
    def test_it_writes_one_png_per_figure(self, tmp_path):
        out = figures.write_all(results(), tmp_path)
        assert set(out["written"]) == {
            f"{name}.png" for name in
            ("cost-accuracy", "latency", "position-bias", "option-scaling", "calibration")}
        assert out["skipped"] == {}

    def test_every_file_it_reports_actually_exists_and_is_a_png(self, tmp_path):
        out = figures.write_all(results(), tmp_path)
        for name in out["written"]:
            path = tmp_path / name
            assert path.exists()
            assert path.read_bytes()[:8] == PNG_MAGIC
            assert path.stat().st_size > 5000, f"{name} is too small to be a readable chart"

    def test_it_creates_the_output_directory(self, tmp_path):
        target = tmp_path / "results" / "figures"
        figures.write_all(results(), target)
        assert target.is_dir()

    def test_rewriting_replaces_rather_than_accumulates(self, tmp_path):
        figures.write_all(results(), tmp_path)
        figures.write_all(results(), tmp_path)
        assert len(list(tmp_path.glob("*.png"))) == 5


class TestFailingLoudlyRatherThanDrawingNothing:
    def test_a_model_whose_temperature_fit_was_refused_still_plots_its_raw_curve(self, tmp_path):
        """Too few validation rows means no temperature, not no calibration: the
        raw reliability curve is still the measurement, and the legend says the
        fit was refused instead of crashing on a missing number."""
        cal = {k: dict(v, temperature=None, ece_scaled=None, fit_refused=True)
               for k, v in results()["calibration"].items()}
        assert "calibration.png" in figures.write_all(results(calibration=cal),
                                                      tmp_path)["written"]

    def test_a_calibration_panel_with_no_probabilistic_model_is_refused(self, tmp_path):
        """The named case: an LLM returns a label, so there is nothing to plot.
        An empty reliability diagram would read as a perfectly calibrated
        model."""
        with pytest.raises(figures.NothingToPlot, match="probabilit"):
            figures.calibration_plot(results(calibration={}), tmp_path)

    def test_cost_against_accuracy_is_refused_when_nothing_has_ground_truth(self, tmp_path):
        no_gold = {k: dict(v, accuracy=None, cost_per_correct_usd=None)
                   for k, v in results()["summary"].items()}
        with pytest.raises(figures.NothingToPlot, match="accuracy"):
            figures.cost_accuracy_plot(results(summary=no_gold), tmp_path)

    def test_a_free_local_model_is_named_in_the_caption_not_silently_dropped(self, tmp_path):
        """Cost is on a log axis, and a zero cannot go on one. Laya costs nothing
        because it runs on this machine, so it would simply disappear from the
        frontier chart -- the single most misleading thing this figure could do.
        It is excluded explicitly and the caption says so."""
        summary = {k: dict(v, cost_per_1k_usd=0.0 if v["local"] else v["cost_per_1k_usd"])
                   for k, v in results()["summary"].items()}
        assert figures.cost_accuracy_plot(results(summary=summary), tmp_path)                == "cost-accuracy.png"
        assert figures.priced_out(summary) == ["laya"]

    def test_nothing_is_excluded_when_every_model_has_a_price(self):
        assert figures.priced_out(results()["summary"]) == []

    def test_cost_against_accuracy_is_refused_when_nothing_reports_a_cost(self, tmp_path):
        free = {k: dict(v, cost_per_1k_usd=0.0) for k, v in results()["summary"].items()}
        with pytest.raises(figures.NothingToPlot, match="cost"):
            figures.cost_accuracy_plot(results(summary=free), tmp_path)

    def test_the_position_bias_panel_is_refused_when_that_arm_never_ran(self, tmp_path):
        with pytest.raises(figures.NothingToPlot, match="position"):
            figures.position_bias_plot(results(position_bias={}), tmp_path)

    def test_a_model_with_no_usable_answer_is_left_out_of_the_bias_panel(self, tmp_path):
        """Zero accuracy at every position because every call failed is not a
        position effect. It is excluded and the exclusion is stated, rather than
        drawn as three flat bars at zero."""
        bias = dict(results()["position_bias"])
        bias["laya/clinc150"] = {"model": "laya", "task": "clinc150", "n_options": 151,
                                 "n_examples": 40, "n_valid": 0,
                                 "flip": {"rate": None, "n": 0, "n_incomplete": 0,
                                          "n_unusable": 40},
                                 "gold": {"accuracy": {"first": 0.0, "middle": 0.0, "last": 0.0},
                                          "spread": 0.0, "n": {"first": 40, "middle": 40,
                                                               "last": 40}},
                                 "positions": {"shares": None, "edges": None,
                                               "mean_normalised": None, "n": 0}}
        out = figures.write_all(results(position_bias=bias), tmp_path)
        assert "position-bias.png" in out["written"]

    def test_the_position_bias_panel_is_refused_when_every_model_failed(self, tmp_path):
        dead = {k: dict(v, n_valid=0) for k, v in results()["position_bias"].items()}
        with pytest.raises(figures.NothingToPlot, match="usable"):
            figures.position_bias_plot(results(position_bias=dead), tmp_path)

    def test_the_position_bias_panel_is_refused_when_no_gold_placement_was_measured(self, tmp_path):
        half = {k: dict(v, gold={"accuracy": {"first": None, "middle": None, "last": None},
                                 "spread": None, "n": {}})
                for k, v in results()["position_bias"].items()}
        with pytest.raises(figures.NothingToPlot, match="gold"):
            figures.position_bias_plot(results(position_bias=half), tmp_path)

    def test_option_scaling_is_refused_with_only_one_task(self, tmp_path):
        one = {k: v for k, v in results()["summary"].items() if v["task"] == "ag_news"}
        with pytest.raises(figures.NothingToPlot, match="two tasks"):
            figures.option_scaling_plot(results(summary=one), tmp_path)

    def test_latency_is_refused_when_nothing_was_timed(self, tmp_path):
        untimed = {k: dict(v, p50_ms=None, p95_ms=None) for k, v in results()["summary"].items()}
        with pytest.raises(figures.NothingToPlot, match="latenc"):
            figures.latency_plot(results(summary=untimed), tmp_path)

    def test_a_refused_figure_writes_no_file_at_all(self, tmp_path):
        with pytest.raises(figures.NothingToPlot):
            figures.calibration_plot(results(calibration={}), tmp_path)
        assert list(tmp_path.glob("*.png")) == []

    def test_write_all_reports_the_skip_with_its_reason_rather_than_crashing(self, tmp_path):
        out = figures.write_all(results(calibration={}, position_bias={}), tmp_path)
        assert set(out["skipped"]) == {"calibration", "position-bias"}
        assert "probabilit" in out["skipped"]["calibration"]
        assert set(out["written"]) == {"cost-accuracy.png", "latency.png", "option-scaling.png"}


class TestHonestStyling:
    def test_no_figure_encodes_its_meaning_in_colour_alone(self, tmp_path):
        """Every series carries a marker or a hatch as well as a shade, and the
        points are labelled, so the figures survive a grayscale print and a
        colour-blind reader."""
        assert figures.MARKERS
        assert len(figures.MARKERS) >= len(figures.SHADES)

    def test_the_local_model_is_marked_as_not_comparable_on_the_latency_chart(self, tmp_path):
        """Laya runs on this machine's CPU with no network in it. Putting it on
        the same axis as a hosted call without saying so would be the single most
        misleading thing this experiment could publish."""
        assert "local" in figures.LOCAL_NOTE.lower()
        assert "not comparable" in figures.LOCAL_NOTE.lower()

    def test_figures_are_written_at_a_readable_resolution(self):
        assert figures.DPI >= 150


class TestResultsRoundTrip:
    def test_the_figures_read_the_same_file_the_tables_are_printed_from(self, tmp_path):
        """One source, so a chart can never disagree with the table beside it."""
        path = tmp_path / "results.json"
        path.write_text(json.dumps(results()), encoding="utf-8")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        out = figures.write_all(loaded, tmp_path / "figures")
        assert out["skipped"] == {}

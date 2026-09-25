"""The site's typed_decisions numbers are exported from the results file, never typed.

Two rules carry most of the weight here. A row that never returned an answer must
not arrive on the page as a zero or as a latency, because a structural refusal is
not a score. And a locally measured latency must stay labelled as one, so nothing
downstream can average it with a hosted round trip.
"""
import json

from typed_decisions.scripts import export_site_data

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def row(model, task="ag_news", **kw):
    base = {
        "model": model, "model_id": f"vendor/{model}", "task": task,
        "transport": "openrouter", "local": False, "n": 300, "n_options": 4,
        "p50_ms": 400.0, "p95_ms": 900.0, "tail_ratio": 2.25,
        "mean_input_tokens": 120.0, "mean_output_tokens": 6.0,
        "cost_per_1k_usd": 0.12, "cost_per_correct_usd": 1.4e-4,
        "accuracy": 0.8, "accuracy_ci": [0.7, 0.9], "valid_rate": 1.0,
        "validity": {"valid": 300}, "retries": 0,
        "returns_probability": model in ("jev", "laya"),
    }
    base.update(kw)
    return base


def bias(model, task="ag_news", **kw):
    base = {
        "model": model, "task": task, "n_options": 4, "n_examples": 40,
        "n_calls": 240, "n_valid": 240,
        "flip": {"rate": 0.05, "n": 40},
        "gold": {"accuracy": {"first": 0.95, "middle": 0.95, "last": 0.9},
                 "spread": 0.05, "n": {"first": 40, "middle": 40, "last": 40}},
        "positions": {"mean_normalised": 0.5, "n": 240},
    }
    base.update(kw)
    return base


REFUSED = row("laya", task="clinc150", transport="local-cpu", local=True, n_options=151,
              p50_ms=None, p95_ms=None, tail_ratio=None, accuracy=None, accuracy_ci=None,
              valid_rate=0.0, validity={"api_error": 300}, mean_input_tokens=None,
              mean_output_tokens=None, cost_per_correct_usd=None, cost_per_1k_usd=0.0,
              returns_probability=False)


def results(**over):
    base = {
        "tag": "full",
        "written_at": "2026-09-25T12:54:53",
        "n_calls": 10120,
        "total_recorded_spend_usd": 0.7245,
        "summary": {
            "jev/ag_news@openrouter": row("jev", p50_ms=230.0),
            "gpt/ag_news@openrouter": row("gpt"),
            "laya/ag_news@local-cpu": row("laya", transport="local-cpu", local=True,
                                          p50_ms=210.0, mean_input_tokens=None),
            "jev/clinc150@openrouter": row("jev", task="clinc150", n_options=151, p50_ms=248.0),
            "gpt/clinc150@openrouter": row("gpt", task="clinc150", n_options=151, p50_ms=800.0),
            "laya/clinc150@local-cpu": REFUSED,
        },
        "position_bias": {
            "jev/ag_news": bias("jev"),
            "laya/clinc150": bias("laya", task="clinc150", n_options=151, n_valid=0,
                                  flip={"rate": None, "n": 0},
                                  gold={"accuracy": {"first": None, "middle": None, "last": None},
                                        "spread": None, "n": {"first": 0, "middle": 0, "last": 0}},
                                  positions={"mean_normalised": None, "n": 0}),
        },
        "calibration": {
            "jev/ag_news": {"model": "jev", "task": "ag_news", "temperature": 1.988,
                            "ece_raw": 0.0705, "ece_scaled": 0.0768,
                            "n_validation": 100, "n_test": 300},
        },
        "option_overhead": {
            "gpt/ag_news": {"n_options": 4, "prompt_tokens_with_options": 144,
                            "option_tokens": 25},
            "deepseek/ag_news": {"n_options": 4, "prompt_tokens_with_options": 110,
                                 "option_tokens": -63},
            "gpt/clinc150": {"n_options": 151, "prompt_tokens_with_options": 1462,
                             "option_tokens": 1374},
            "laya/ag_news": {"n_options": 4, "prompt_tokens_with_options": None,
                             "option_tokens": None},
        },
        "tasks": {
            "ag_news": {"dataset": "fancyzhx/ag_news", "licence": "unknown", "n_examples": 300,
                        "n_options": 4, "n_validation": 100, "bias_subset": 40,
                        "n_random_orders": 3},
            "clinc150": {"dataset": "clinc/clinc_oos", "licence": "CC-BY-3.0", "n_examples": 300,
                         "n_options": 151, "n_validation": 100, "bias_subset": 40,
                         "n_random_orders": 3},
        },
        "model_resolution": {"laya": {"cfg": {"head_max_len": 192, "max_len": 512}}},
    }
    base.update(over)
    return base


def find(rows, model, task):
    return next(r for r in rows if r["model"] == model and r["task"] == task)


def test_a_refused_row_exports_no_accuracy_and_no_latency():
    # 300 calls that never returned is a transport failure, not a score of zero.
    laya = find(export_site_data.build(results())["models"], "laya", "clinc150")
    assert laya["accuracy"] is None
    assert laya["p50_ms"] is None
    assert laya["valid_rate"] == 0.0
    assert laya["validity"] == {"api_error": 300}


def test_a_locally_measured_latency_stays_labelled_as_one():
    laya = find(export_site_data.build(results())["models"], "laya", "ag_news")
    assert laya["local"] is True
    assert laya["transport"] == "local-cpu"


def test_a_local_row_sorts_below_the_hosted_ones_even_when_it_is_faster():
    # laya's 210 ms beats jev's 230 ms and is measured on CPU with no network in
    # it. Ordering it first would read as a speed ranking that does not exist.
    rows = [r for r in export_site_data.build(results())["models"] if r["task"] == "ag_news"]
    assert [r["model"] for r in rows] == ["jev", "gpt", "laya"]


def test_latency_rise_needs_both_option_counts_measured():
    rise = export_site_data.build(results())["latency_rise"]
    assert [r["model"] for r in rise] == ["jev", "gpt"]
    jev = next(r for r in rise if r["model"] == "jev")
    assert round(jev["rise"], 4) == round(248.0 / 230.0 - 1, 4)
    assert "laya" not in [r["model"] for r in rise]


def test_an_impossible_option_overhead_is_excluded_and_named():
    share = next(s for s in export_site_data.build(results())["option_share"]
                 if s["task"] == "ag_news")
    # deepseek billed FEWER tokens with the option list than without it. That is
    # the provider's accounting and no share can be computed from it.
    assert share["unusable"] == ["deepseek/ag_news"]
    assert share["n_models"] == 1
    assert round(share["max_share"], 4) == round(25 / 144, 4)


def test_a_bias_row_with_nothing_usable_keeps_its_nulls():
    laya = find(export_site_data.build(results())["position_bias"], "laya", "clinc150")
    assert laya["flip_rate"] is None
    assert laya["spread"] is None
    assert laya["n_valid"] == 0
    assert laya["n_calls"] == 240


def test_the_probability_split_is_read_off_the_records():
    prob = export_site_data.build(results())["probability"]
    # laya returns a distribution on the task it can answer, so one refused task
    # must not move it into the label-only column.
    assert prob["returns"] == ["jev", "laya"]
    assert prob["label_only"] == ["gpt"]


def test_figures_are_copied_with_the_size_the_page_reserves(tmp_path):
    source = tmp_path / "figures"
    source.mkdir()
    # 1x1 PNG: enough for the IHDR the exporter reads its dimensions from.
    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
        "890000000a49444154789c6300010000050001"
        "0d0a2db40000000049454e44ae426082")
    for name in ("latency.png", "position-bias.png"):
        (source / name).write_bytes(png)
    out = tmp_path / "public"
    figures = export_site_data.copy_figures(source, out)
    assert figures["latency"]["src"] == "/typed-decisions-latency.png"
    assert figures["latency"]["width"] == 1 and figures["latency"]["height"] == 1
    assert (out / "typed-decisions-position-bias.png").read_bytes()[:8] == PNG_MAGIC


def test_the_committed_results_file_still_says_what_the_page_says():
    """The one test that fails if the run is redone and the page goes stale."""
    built = export_site_data.build(json.loads(export_site_data.RESULTS.read_text("utf-8")))
    jev4 = find(built["models"], "jev", "ag_news")
    jev151 = find(built["models"], "jev", "clinc150")
    assert round(jev4["p50_ms"]) == 230 and round(jev151["p50_ms"]) == 248
    assert jev4["accuracy"] == 0.89
    # Fastest on both tasks, and it wins on accuracy nowhere. Laya's median is
    # lower still on ag_news and is measured on local CPU, so it is excluded from
    # the speed ranking and kept in the accuracy one: accuracy does not depend on
    # the transport and a latency does.
    for task in ("ag_news", "clinc150"):
        hosted = [r for r in built["models"]
                  if r["task"] == task and r["p50_ms"] and not r["local"]]
        scored = [r for r in built["models"] if r["task"] == task and r["accuracy"]]
        assert min(hosted, key=lambda r: r["p50_ms"])["model"] == "jev"
        assert max(scored, key=lambda r: r["accuracy"])["model"] != "jev"
    assert find(built["models"], "laya", "ag_news")["local"] is True
    laya = find(built["models"], "laya", "clinc150")
    assert laya["accuracy"] is None and laya["valid_rate"] == 0.0
    assert built["laya_cfg"]["head_max_len"] == 192
    assert len(built["position_bias"]) == 18

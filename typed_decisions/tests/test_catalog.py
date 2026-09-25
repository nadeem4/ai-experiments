"""Which model each family resolves to, and why.

OpenRouter's catalog lists 460 models; this account can call a fraction of them.
Selection is therefore two steps -- a pure ranking over the catalog, then a live
probe -- and only the ranking is testable without a network, which is what this
file covers. The probe is injected, so the resolution loop is testable too.

The decision models are not in the default listing at all and are fetched from a
separate query, which is why `JEV_ID` is pinned rather than ranked.
"""
import calendar
import datetime

import pytest

from typed_decisions import catalog


def created(year, month):
    return calendar.timegm(datetime.datetime(year, month, 1).timetuple())


def entry(model_id, year, month, prompt, params=("structured_outputs", "temperature"),
          modality="text"):
    return {
        "id": model_id,
        "canonical_slug": f"{model_id}-{year}{month:02d}01",
        "created": created(year, month),
        "architecture": {"output_modalities": [modality]},
        "pricing": {"prompt": str(prompt), "completion": "0.0000005"},
        "supported_parameters": list(params),
    }


CATALOG = [
    # current generation, and deliberately NOT the cheapest: the old ranking
    # picked on price alone and so resolved this family to a 2025 model.
    entry("openai/gpt-6-luna", 2026, 9, 0.0000002, params=("structured_outputs",)),
    entry("openai/gpt-6-sol", 2026, 9, 0.000002, params=("structured_outputs",)),
    entry("openai/gpt-6-luna:batch", 2026, 9, 0.0000001, params=("structured_outputs",)),
    entry("openai/gpt-6-luna-pro", 2026, 9, 0.0000001, params=("structured_outputs",)),
    entry("openai/gpt-4.1-nano", 2025, 4, 0.0000001),
    entry("openai/gpt-3.5-turbo", 2023, 3, 0.00000001, params=("tools", "temperature")),
    entry("anthropic/claude-sonnet-5", 2026, 6, 0.000002, params=("structured_outputs",)),
    entry("anthropic/claude-haiku-4.5", 2025, 10, 0.000001),
    entry("google/gemini-3.8-flash", 2026, 9, 0.00000075),
    entry("google/gemini-2.5-flash-lite", 2025, 7, 0.0000001),
    entry("google/gemini-3.1-flash-image", 2026, 2, 0.0000001, modality="image"),
    entry("google/gemma-4-31b-it", 2026, 4, 0.00000009),
    entry("google/gemma-4-26b-a4b-it", 2026, 4, 0.00000009),
    entry("google/gemma-3-4b-it", 2025, 3, 0.00000005),
    entry("typesafe/jev-1.13", 2026, 9, 0.000000042, params=(), modality="decisions"),
]


def ids(prefix):
    return [m["id"] for m in catalog.candidates(CATALOG, prefix)]


def test_candidates_lead_with_the_current_generation_not_the_cheapest_model():
    """The rule that changed with the OpenRouter migration. Ranking on price alone
    resolved every family to a 2025 model even where the current one was reachable,
    which is exactly the silent substitution the protocol forbids."""
    assert ids("openai/gpt-")[0] == "openai/gpt-6-luna"
    assert ids("google/gemini-")[0] == "google/gemini-3.8-flash"


def test_within_one_generation_the_cheapest_tier_still_wins():
    """`cheapest tier` was always the intent -- the nano/lite/flash class -- and it
    still is, applied inside the current generation rather than across releases."""
    assert ids("openai/gpt-")[:2] == ["openai/gpt-6-luna", "openai/gpt-6-sol"]


def test_an_older_generation_is_kept_as_a_fallback_below_the_current_one():
    """It is a fallback, not a substitute: the ladder is recorded so the report can
    say the current model was refused and an older one carried the family."""
    assert ids("openai/gpt-")[-1] == "openai/gpt-4.1-nano"


def test_a_model_that_refuses_a_temperature_is_still_a_candidate():
    """The bug this migration fixed. The current GPT and Claude tiers do not accept
    a temperature, and requiring one silently excluded every current-generation
    model. Fairness rule 3 already covers it: temperature 0 where the model takes
    one, null where it does not, recorded either way."""
    assert "openai/gpt-6-luna" in ids("openai/gpt-")
    assert catalog.accepts_temperature(catalog.find(CATALOG, "openai/gpt-6-luna")) is False
    assert catalog.accepts_temperature(catalog.find(CATALOG, "openai/gpt-4.1-nano")) is True


def test_candidates_drop_models_that_cannot_be_asked_in_structured_mode():
    assert "openai/gpt-3.5-turbo" not in ids("openai/gpt-")  # no structured_outputs, however cheap


def test_candidates_drop_the_batch_variants():
    """`:batch` is an asynchronous queue at half price. It is a different product
    with a different latency, so averaging it into a latency table would be a lie."""
    assert "openai/gpt-6-luna:batch" not in ids("openai/gpt-")


def test_candidates_drop_denied_variants_however_cheap():
    assert "openai/gpt-6-luna-pro" not in ids("openai/gpt-")


def test_candidates_drop_everything_that_is_not_a_text_model():
    assert ids("google/gemini-") == ["google/gemini-3.8-flash", "google/gemini-2.5-flash-lite"]


def test_a_decision_model_is_not_a_candidate_for_an_llm_family():
    """Jev answers a typed question on a different route, so it can never be picked
    as though it were one of the chat models."""
    assert "typesafe/jev-1.13" not in ids("typesafe/")


def test_gemma_is_its_own_family_and_does_not_leak_into_gemini():
    assert ids("google/gemma-")[0] in ("google/gemma-4-31b-it", "google/gemma-4-26b-a4b-it")
    assert "google/gemini-3.8-flash" not in ids("google/gemma-")


def test_resolve_takes_the_first_candidate_the_probe_can_actually_reach():
    reachable = {"openai/gpt-4.1-nano"}
    picked, tried = catalog.resolve(CATALOG, "openai/gpt-", probe=lambda i: i in reachable)
    assert picked["id"] == "openai/gpt-4.1-nano"
    assert tried[0] == ("openai/gpt-6-luna", False)
    assert tried[-1] == ("openai/gpt-4.1-nano", True)


def test_resolve_records_every_id_it_was_refused_so_the_report_can_say_so():
    """`do not silently substitute an older model` is only enforceable if the
    refusals are on the record."""
    _, tried = catalog.resolve(CATALOG, "openai/gpt-", probe=lambda i: i == "openai/gpt-4.1-nano")
    assert [t for t, ok in tried if not ok] == ["openai/gpt-6-luna", "openai/gpt-6-sol"]


def test_resolve_reports_a_family_it_could_not_reach_rather_than_substituting():
    picked, tried = catalog.resolve(CATALOG, "openai/gpt-", probe=lambda i: False)
    assert picked is None
    assert all(ok is False for _, ok in tried)


def test_generation_buckets_by_release_month():
    """Two models released the same month are the same generation and are then
    ranked on price; a month apart is a different rung on the ladder."""
    a = catalog.find(CATALOG, "google/gemma-4-31b-it")
    b = catalog.find(CATALOG, "google/gemma-4-26b-a4b-it")
    c = catalog.find(CATALOG, "google/gemma-3-4b-it")
    assert catalog.generation(a) == catalog.generation(b)
    assert catalog.generation(a) > catalog.generation(c)


def test_price_reads_openrouters_prompt_field():
    """OpenRouter spells it `prompt`/`completion`, the Vercel gateway spelled it
    `input`/`output`. Reading the wrong key silently prices everything at zero."""
    assert catalog.price(catalog.find(CATALOG, "openai/gpt-6-luna")) == pytest.approx(2e-07)


def test_version_records_the_dated_canonical_slug_not_just_the_alias():
    """`typesafe/jev-1.13` floats; `typesafe/jev-1.13-20260917` does not. A
    benchmark records the one that cannot move under it."""
    model = catalog.find(CATALOG, "openai/gpt-6-luna")
    v = catalog.version(model)
    assert v["id"] == "openai/gpt-6-luna"
    assert v["canonical_slug"] == "openai/gpt-6-luna-20260901"
    assert v["released"] == "2026-09"
    assert v["pricing"] == model["pricing"]  # carried through verbatim, never reformatted


def test_accepts_reasoning_effort_is_read_from_the_catalog_not_assumed():
    """The current tiers are reasoning models whose thinking tokens can eat a small
    answer budget. The flag is only sent to models that advertise it -- the same
    shape as the temperature rule."""
    reasoner = entry("x/y", 2026, 9, 0.1, params=("structured_outputs", "reasoning_effort"))
    assert catalog.accepts_reasoning_effort(reasoner) is True
    assert catalog.accepts_reasoning_effort(catalog.find(CATALOG, "google/gemma-4-31b-it")) is False


def test_the_families_are_the_slate_and_nothing_else():
    """`FAMILIES` is the older name for the same mapping, kept so the resolution
    code below reads as it always did."""
    assert catalog.FAMILIES is catalog.HOSTED
    assert set(catalog.FAMILIES) == {"glm", "phi", "gemma", "deepseek", "llama", "qwen", "gpt"}


def test_jev_is_pinned_to_an_explicit_dated_version():
    """The alias `~typesafe/jev-latest` and the floating `typesafe/jev-1.13` both
    move. A benchmark pins the version."""
    assert catalog.JEV_ID == "typesafe/jev-1.13-20260917"


@pytest.mark.parametrize("denied", ["openai/gpt-5-fast", "openai/gpt-5.1-codex",
                                    "openai/gpt-6-luna-pro", "openai/gpt-6-luna:batch"])
def test_the_deny_list_is_a_substring_match_on_the_id(denied):
    assert catalog.denied(denied)


# --- the flagship tiers, denied so "cheapest in the current generation" means it


@pytest.mark.parametrize("flagship", ["qwen/qwen3.8-max-0902", "qwen/qwen3.8-max-prime",
                                      "openai/gpt-6-astra-max"])
def test_the_flagship_tiers_are_denied(flagship):
    """Every family is represented by the cheap tier of its current generation.
    Left in, Qwen resolved to a $2/M `-max` and a $4/M `-prime`, which is not the
    tier anybody classifies support intents with."""
    assert catalog.denied(flagship)


@pytest.mark.parametrize("free", ["qwen/qwen3.8-27b:free", "z-ai/glm-5.3-flash:free"])
def test_the_free_mirrors_are_denied(free):
    """A `:free` mirror is a rate-limited queue at a different latency, the same
    objection as `:batch`. It must never appear in a latency table."""
    assert catalog.denied(free)


@pytest.mark.parametrize("kept", ["qwen/qwen3.8-flash", "z-ai/glm-5.3-flash",
                                  "deepseek/deepseek-v4.1-flash", "microsoft/phi-4",
                                  "google/gemma-4-26b-a4b-it", "meta-llama/llama-4-scout",
                                  "openai/gpt-6-luna"])
def test_the_slate_itself_survives_the_deny_list(kept):
    """A deny list wide enough to remove the flagships and narrow enough to keep
    the models the experiment actually runs."""
    assert not catalog.denied(kept)


# =============================================================================
# The slate: adding a model is configuration, not code
# =============================================================================
#
# The model registry: adding a model is configuration, not code.
#
# The requirement behind this file is that the slate grows. A hosted model should
# be one line mapping a CLI name to a catalogue prefix (or an exact id on the
# command line); a second local decision model should be one new adapter module and
# one registry line, with nothing in `run.py` or `report.py` needing to know.


class TestTheSlate:
    def test_the_hosted_families_are_the_ones_the_protocol_names(self):
        assert set(catalog.HOSTED) == {"glm", "phi", "gemma", "deepseek", "llama", "qwen", "gpt"}

    def test_gemini_is_not_on_the_slate(self):
        """Excluded deliberately: it spent 318 completion tokens on a one-word
        answer and would have been most of the bill."""
        assert "gemini" not in catalog.ALL

    def test_claude_is_not_on_the_slate(self):
        """The only current-generation Claude this account reaches is an Opus
        tier, and nobody deploys a flagship for intent classification."""
        assert "claude" not in catalog.ALL

    def test_the_decision_models_are_their_own_kind(self):
        assert set(catalog.DECISION) == {"jev"}
        assert set(catalog.LOCAL) == {"laya"}

    def test_jev_is_pinned_to_a_dated_build_not_a_floating_id(self):
        pinned = catalog.DECISION["jev"]["model_id"]
        assert pinned == "typesafe/jev-1.13-20260917"
        assert not pinned.startswith("~")

    def test_every_name_appears_exactly_once_across_the_three_kinds(self):
        names = list(catalog.HOSTED) + list(catalog.DECISION) + list(catalog.LOCAL)
        assert len(names) == len(set(names))
        assert set(names) == set(catalog.ALL)


class TestProbabilityCapability:
    def test_decision_models_return_a_distribution_over_every_option(self):
        assert catalog.returns_probability("jev") is True
        assert catalog.returns_probability("laya") is True

    def test_an_llm_in_structured_mode_returns_a_label_and_nothing_else(self):
        """Without a probability you cannot threshold, so 'ask a human when
        unsure' is not available at all. It is a column, not a footnote."""
        for name in catalog.HOSTED:
            assert catalog.returns_probability(name) is False


class TestParsingTheCliArgument:
    def test_all_selects_every_registered_model(self):
        assert list(catalog.select("all")) == list(catalog.ALL)

    def test_a_subset_selects_only_those(self):
        assert catalog.select("glm,phi,jev") == {"glm": None, "phi": None, "jev": None}

    def test_a_name_can_be_pinned_to_an_exact_id(self):
        assert catalog.select("glm=z-ai/glm-5.3-flash")["glm"] == "z-ai/glm-5.3-flash"

    def test_whitespace_around_a_name_is_tolerated(self):
        assert set(catalog.select("glm, phi ")) == {"glm", "phi"}

    def test_an_unknown_name_is_refused_and_the_message_lists_the_known_ones(self):
        with pytest.raises(SystemExit, match="gemini"):
            catalog.select("gemini")
        with pytest.raises(SystemExit, match="glm"):
            catalog.select("gemini")


class TestLocalAdapters:
    def test_a_local_model_is_one_registry_line_naming_an_adapter(self):
        """`module:Class`. Adding a second local decision model is a new module
        and a line here; nothing in the runner changes."""
        assert catalog.LOCAL["laya"]["adapter"] == "typed_decisions.laya_local:LayaModel"

    def test_the_adapter_is_resolved_by_dotted_path(self):
        from typed_decisions.laya_local import LayaModel

        assert catalog.load_adapter("typed_decisions.laya_local:LayaModel") is LayaModel

    def test_a_missing_adapter_fails_with_the_path_in_the_message(self):
        with pytest.raises(SystemExit, match="nope.module:Thing"):
            catalog.load_adapter("nope.module:Thing")


class TestKinds:
    def test_kind_of_names_the_transport_family(self):
        assert catalog.kind_of("gpt") == "hosted"
        assert catalog.kind_of("jev") == "decision"
        assert catalog.kind_of("laya") == "local"

    def test_only_the_local_models_are_grouped_apart_in_tables(self):
        """A local CPU number and a network round trip are not a latency
        comparison, so the grouping is mechanical rather than remembered."""
        assert catalog.is_local("laya") is True
        assert catalog.is_local("jev") is False
        assert catalog.is_local("gpt") is False

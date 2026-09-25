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

from decision_cost import catalog


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


def test_families_cover_the_four_hosted_llm_families_the_experiment_wants():
    assert set(catalog.FAMILIES) == {"gpt", "claude", "gemini", "gemma"}


def test_jev_is_pinned_to_an_explicit_dated_version():
    """The alias `~typesafe/jev-latest` and the floating `typesafe/jev-1.13` both
    move. A benchmark pins the version."""
    assert catalog.JEV_ID == "typesafe/jev-1.13-20260917"


@pytest.mark.parametrize("denied", ["openai/gpt-5-fast", "openai/gpt-5.1-codex",
                                    "openai/gpt-6-luna-pro", "openai/gpt-6-luna:batch"])
def test_the_deny_list_is_a_substring_match_on_the_id(denied):
    assert catalog.denied(denied)

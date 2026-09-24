"""Which model each family resolves to, and why.

The gateway's catalog lists 390 models; this account can call a fraction of them.
Selection is therefore two steps -- a pure ranking over the catalog, then a live
probe -- and only the ranking is testable without a network, which is what this
file covers. The probe is injected, so the resolution loop is testable too.
"""
import pytest

from latency import catalog

CATALOG = [
    {"id": "openai/gpt-6-luna", "type": "language", "released": 2000,
     "supported_parameters": ["structured_outputs", "response_format"], "temperature": True,
     "pricing": {"input": "0.0000001", "output": "0.0000005"}},
    {"id": "openai/gpt-4.1-nano", "type": "language", "released": 1000,
     "supported_parameters": ["structured_outputs", "response_format"], "temperature": True,
     "pricing": {"input": "0.0000001", "output": "0.0000004"}},
    {"id": "openai/gpt-4.1", "type": "language", "released": 1000,
     "supported_parameters": ["structured_outputs"], "temperature": True,
     "pricing": {"input": "0.000002", "output": "0.000008"}},
    {"id": "openai/gpt-5-pro", "type": "language", "released": 3000,
     "supported_parameters": ["structured_outputs"], "temperature": True,
     "pricing": {"input": "0.0000000001", "output": "0"}},
    {"id": "openai/gpt-3.5-turbo", "type": "language", "released": 500,
     "supported_parameters": ["tools"], "temperature": True,
     "pricing": {"input": "0.00000000001", "output": "0"}},
    {"id": "google/gemini-3.1-flash-image", "type": "image", "released": 2000,
     "supported_parameters": ["structured_outputs"], "temperature": True,
     "pricing": {"input": "0.0000001", "output": "0"}},
    {"id": "google/gemini-2.5-flash-lite", "type": "language", "released": 900,
     "supported_parameters": ["structured_outputs"], "temperature": True,
     "pricing": {"input": "0.0000001", "output": "0.0000004"}},
    {"id": "google/gemma-4-31b-it", "type": "language", "released": 900,
     "supported_parameters": ["structured_outputs"], "temperature": True,
     "pricing": {"input": "0.00000014", "output": "0.0000004"}},
    {"id": "anthropic/claude-opus-5.5", "type": "language", "released": 3000,
     "supported_parameters": ["tools"], "temperature": False,
     "pricing": {"input": "0.000004", "output": "0.00002"}},
]


def test_candidates_are_cheapest_first_then_newest():
    """Cheapest first is the rule: this experiment measures what a decision costs,
    so each family is represented by the cheapest tier that can still be asked in
    structured mode. Equal price breaks towards the newer model."""
    ids = [m["id"] for m in catalog.candidates(CATALOG, "openai/gpt-")]
    assert ids[:2] == ["openai/gpt-6-luna", "openai/gpt-4.1-nano"]


def test_candidates_drop_models_that_cannot_be_asked_in_structured_mode():
    ids = [m["id"] for m in catalog.candidates(CATALOG, "openai/gpt-")]
    assert "openai/gpt-3.5-turbo" not in ids  # no structured_outputs, however cheap


def test_candidates_drop_denied_variants_however_cheap():
    """`-pro`, `-fast`, `codex` and the rest are different products at different
    prices for the same family; including them would make the pick arbitrary."""
    ids = [m["id"] for m in catalog.candidates(CATALOG, "openai/gpt-")]
    assert "openai/gpt-5-pro" not in ids


def test_candidates_drop_everything_that_is_not_a_language_model():
    ids = [m["id"] for m in catalog.candidates(CATALOG, "google/gemini-")]
    assert ids == ["google/gemini-2.5-flash-lite"]


def test_gemma_is_its_own_family_and_does_not_leak_into_gemini():
    assert [m["id"] for m in catalog.candidates(CATALOG, "google/gemma-")] == ["google/gemma-4-31b-it"]


def test_resolve_takes_the_first_candidate_the_probe_can_actually_reach():
    """The catalog lists models this account is not entitled to. Only a live call
    tells them apart, so the first reachable candidate wins."""
    reachable = {"openai/gpt-4.1-nano"}
    picked, tried = catalog.resolve(CATALOG, "openai/gpt-", probe=lambda i: i in reachable)
    assert picked["id"] == "openai/gpt-4.1-nano"
    assert tried == [("openai/gpt-6-luna", False), ("openai/gpt-4.1-nano", True)]


def test_resolve_reports_a_family_it_could_not_reach_rather_than_substituting():
    picked, tried = catalog.resolve(CATALOG, "anthropic/claude-", probe=lambda i: True)
    assert picked is None  # nothing in this catalog survives the filters
    assert tried == []


def test_version_records_what_the_gateway_says_about_the_model():
    m = dict(CATALOG[0], name="GPT-6 Luna", context_window=1050000, knowledge="2026-04")
    assert catalog.version(m) == {
        "id": "openai/gpt-6-luna", "name": "GPT-6 Luna", "released": 2000,
        "context_window": 1050000, "knowledge": "2026-04",
        "pricing": {"input": "0.0000001", "output": "0.0000005"},
    }


def test_families_cover_the_four_hosted_llm_families_the_experiment_wants():
    assert set(catalog.FAMILIES) == {"gpt", "claude", "gemini", "gemma"}


@pytest.mark.parametrize("denied", ["openai/gpt-5-fast", "openai/gpt-5.1-codex", "openai/gpt-5-pro"])
def test_the_deny_list_is_a_substring_match_on_the_id(denied):
    assert catalog.denied(denied)

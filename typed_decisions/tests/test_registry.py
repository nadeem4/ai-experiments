"""The model registry: adding a model is configuration, not code.

The requirement behind this file is that the slate grows. A hosted model should
be one line mapping a CLI name to a catalogue prefix (or an exact id on the
command line); a second local decision model should be one new adapter module and
one registry line, with nothing in `run.py` or `report.py` needing to know.
"""
import pytest

from typed_decisions import registry


class TestTheSlate:
    def test_the_hosted_families_are_the_ones_the_protocol_names(self):
        assert set(registry.HOSTED) == {"glm", "phi", "gemma", "deepseek", "llama", "qwen", "gpt"}

    def test_gemini_is_not_on_the_slate(self):
        """Excluded deliberately: it spent 318 completion tokens on a one-word
        answer and would have been most of the bill."""
        assert "gemini" not in registry.ALL

    def test_claude_is_not_on_the_slate(self):
        """The only current-generation Claude this account reaches is an Opus
        tier, and nobody deploys a flagship for intent classification."""
        assert "claude" not in registry.ALL

    def test_the_decision_models_are_their_own_kind(self):
        assert set(registry.DECISION) == {"jev"}
        assert set(registry.LOCAL) == {"laya"}

    def test_jev_is_pinned_to_a_dated_build_not_a_floating_id(self):
        pinned = registry.DECISION["jev"]["model_id"]
        assert pinned == "typesafe/jev-1.13-20260917"
        assert not pinned.startswith("~")

    def test_every_name_appears_exactly_once_across_the_three_kinds(self):
        names = list(registry.HOSTED) + list(registry.DECISION) + list(registry.LOCAL)
        assert len(names) == len(set(names))
        assert set(names) == set(registry.ALL)


class TestProbabilityCapability:
    def test_decision_models_return_a_distribution_over_every_option(self):
        assert registry.returns_probability("jev") is True
        assert registry.returns_probability("laya") is True

    def test_an_llm_in_structured_mode_returns_a_label_and_nothing_else(self):
        """Without a probability you cannot threshold, so 'ask a human when
        unsure' is not available at all. It is a column, not a footnote."""
        for name in registry.HOSTED:
            assert registry.returns_probability(name) is False


class TestParsingTheCliArgument:
    def test_all_selects_every_registered_model(self):
        assert list(registry.parse("all")) == list(registry.ALL)

    def test_a_subset_selects_only_those(self):
        assert registry.parse("glm,phi,jev") == {"glm": None, "phi": None, "jev": None}

    def test_a_name_can_be_pinned_to_an_exact_id(self):
        assert registry.parse("glm=z-ai/glm-5.3-flash")["glm"] == "z-ai/glm-5.3-flash"

    def test_whitespace_around_a_name_is_tolerated(self):
        assert set(registry.parse("glm, phi ")) == {"glm", "phi"}

    def test_an_unknown_name_is_refused_and_the_message_lists_the_known_ones(self):
        with pytest.raises(SystemExit, match="gemini"):
            registry.parse("gemini")
        with pytest.raises(SystemExit, match="glm"):
            registry.parse("gemini")


class TestLocalAdapters:
    def test_a_local_model_is_one_registry_line_naming_an_adapter(self):
        """`module:Class`. Adding a second local decision model is a new module
        and a line here; nothing in the runner changes."""
        assert registry.LOCAL["laya"]["adapter"] == "typed_decisions.laya_local:LayaModel"

    def test_the_adapter_is_resolved_by_dotted_path(self):
        from typed_decisions.laya_local import LayaModel

        assert registry.load_adapter("typed_decisions.laya_local:LayaModel") is LayaModel

    def test_a_missing_adapter_fails_with_the_path_in_the_message(self):
        with pytest.raises(SystemExit, match="nope.module:Thing"):
            registry.load_adapter("nope.module:Thing")


class TestKinds:
    def test_kind_of_names_the_transport_family(self):
        assert registry.kind_of("gpt") == "hosted"
        assert registry.kind_of("jev") == "decision"
        assert registry.kind_of("laya") == "local"

    def test_only_the_local_models_are_grouped_apart_in_tables(self):
        """A local CPU number and a network round trip are not a latency
        comparison, so the grouping is mechanical rather than remembered."""
        assert registry.is_local("laya") is True
        assert registry.is_local("jev") is False
        assert registry.is_local("gpt") is False

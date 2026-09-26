"""`cli dataset <name>` writes what the data actually is.

Written by code from the real files, because a dataset description typed by hand
is one more fact that has to be remembered, and this repository has already
published two that were not.
"""
import pytest

from cli import dataset

PROFILE = {
    "name": "BANKING77",
    "source": "PolyAI-LDN/task-specific-datasets",
    "url": "https://github.com/PolyAI-LDN/task-specific-datasets",
    "licence": "CC-BY-4.0",
    "what_it_is": "Customer banking queries, each labelled with one intent.",
    "splits": {"train": 10003, "test": 3080},
    "used": "test",
    "classes": ["card_arrival", "card_linking", "exchange_rate"],
    "rows": [
        {"text": "I am still waiting on my card?", "category": "card_arrival"},
        {"text": "What can I do if my card still hasn't arrived?", "category": "card_arrival"},
    ],
}


class TestRendering:
    def test_it_names_the_source_and_links_it(self):
        out = dataset.render([PROFILE])
        assert "BANKING77" in out
        assert "https://github.com/PolyAI-LDN/task-specific-datasets" in out

    def test_it_shows_real_rows_not_a_description_of_them(self):
        out = dataset.render([PROFILE])
        assert "I am still waiting on my card?" in out

    def test_it_reports_every_split_and_which_one_is_used(self):
        out = dataset.render([PROFILE])
        assert "10,003" in out and "3,080" in out
        assert "test" in out

    def test_a_long_class_list_is_counted_and_sampled_not_dumped(self):
        """151 intent names down the page is not information."""
        many = {**PROFILE, "classes": [f"intent_{i}" for i in range(151)]}
        out = dataset.render([many])
        assert "151" in out
        assert "intent_0" in out
        assert "intent_150" not in out, "a sample, not the whole list"

    def test_two_datasets_render_as_two_sections(self):
        out = dataset.render([PROFILE, {**PROFILE, "name": "AG News"}])
        assert out.count("## ") == 2

    def test_it_says_when_a_licence_is_not_known(self):
        out = dataset.render([{**PROFILE, "licence": None}])
        assert "not recorded" in out.lower()


class TestEveryExperimentCanDescribeItsData:
    @pytest.mark.parametrize("name", ("rerank", "banking77", "typed_decisions"))
    def test_it_exposes_a_dataset_function(self, name):
        import cli

        assert callable(getattr(cli.load(name), "dataset", None)), \
            f"{name} cannot describe its own data"

"""The dataset card `uv run cli dataset emeans` writes."""
import pytest

from emeans import embed
from tests.test_emeans_embed import Stub, sample


def test_dataset_card_reports_counts_and_revision(tmp_path):
    embed.run(Stub(), sample(), tmp_path, revision="abc")
    from emeans import experiment

    card = experiment.dataset(tmp_path)[0]
    assert card["extra"]["Per-month counts"].startswith("2018-01: 2")
    assert "abc" in card["extra"]["Model revision"]
    assert card["splits"]["docs"] == 3
    for key in ("name", "what_it_is", "source", "url", "licence", "splits", "used", "rows"):
        assert key in card


def test_dataset_card_without_vectors_is_a_clear_exit(tmp_path):
    from emeans import experiment

    with pytest.raises(SystemExit):
        experiment.dataset(tmp_path)

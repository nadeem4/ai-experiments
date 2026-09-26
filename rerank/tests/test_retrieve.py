"""The first stage: hybrid retrieval, and the fusion that combines the two halves.

This is not what the experiment measures -- the scorer is -- but it decides what
the scorers are given, so it is held fixed and pinned.
"""
import pytest

from rerank import retrieve


class TestReciprocalRankFusion:
    """Rank-based, not score-based: BM25 scores and cosine similarities are not
    on the same scale and normalising them is a tuning knob nobody should have
    to choose."""

    def test_a_document_both_halves_rank_first_wins(self):
        fused = retrieve.fuse({"a": ["d1", "d2"], "b": ["d1", "d3"]})
        assert fused[0] == "d1"

    def test_a_document_only_one_half_found_still_appears(self):
        fused = retrieve.fuse({"a": ["d1"], "b": ["d2"]})
        assert set(fused) == {"d1", "d2"}

    def test_rank_decides_not_score_magnitude(self):
        """BM25 can return 30.0 and cosine 0.31 for the same quality of match."""
        fused = retrieve.fuse({"lexical": ["x", "y"], "dense": ["y", "x"]})
        assert set(fused) == {"x", "y"}, "both survive; neither scale dominates"

    def test_ties_are_broken_deterministically(self):
        once = retrieve.fuse({"a": ["p", "q"], "b": ["q", "p"]})
        again = retrieve.fuse({"a": ["p", "q"], "b": ["q", "p"]})
        assert once == again

    def test_k_damps_how_much_the_top_rank_dominates(self):
        """A document ranked first by one half and absent from the other should
        not automatically beat one ranked second by both."""
        fused = retrieve.fuse({"a": ["solo", "both"], "b": ["other", "both"]}, k=1)
        assert fused[0] == "both"


class TestWhatIsRecorded:
    """A candidate set that cannot be reproduced makes every number below it
    unreproducible too, so the retriever's settings go in the run config."""

    def test_the_settings_name_the_model_and_the_fusion(self):
        settings = retrieve.settings()
        assert settings["encoder"].startswith("sentence-transformers/")
        assert settings["fusion"] == "reciprocal-rank"
        assert isinstance(settings["rrf_k"], int)
        assert settings["depth"] >= 20, "each half must go deeper than the final top-k"

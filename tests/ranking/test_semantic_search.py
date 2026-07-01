"""Tests the semantic search module."""

from unittest.mock import Mock

import numpy as np
import pytest

from research_paper_intelligence.retrieval.semantic_search import (
    semantic_search,
)


class TestSemanticSearch:
    """Tests the semantic search function."""

    def test_returns_candidate_positions_and_scores(self) -> None:
        """Tests that it returns the candidate positions and scores."""
        model = Mock()
        model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3]],
            dtype=np.float32,
        )

        index = Mock()
        index.ntotal = 3
        index.search.return_value = (
            np.array([[0.90, 0.75]], dtype=np.float32),
            np.array([[2, 0]], dtype=np.int64),
        )

        positions, scores = semantic_search(
            query="  machine learning  ",
            model=model,
            index=index,
            candidate_top_k=2,
        )

        np.testing.assert_array_equal(
            positions,
            np.array([2, 0], dtype=np.int64),
        )
        np.testing.assert_allclose(
            scores,
            np.array([0.90, 0.75], dtype=np.float64),
        )

        model.encode.assert_called_once_with(
            ["machine learning"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        query_embedding, search_k = index.search.call_args.args

        assert query_embedding.dtype == np.float32
        assert query_embedding.flags["C_CONTIGUOUS"]
        assert search_k == 2

    @pytest.mark.parametrize(
        ("query", "candidate_top_k", "message"),
        [
            ("   ", 2, "Query cannot be empty."),
            (
                "machine learning",
                0,
                "candidate_top_k must be a positive integer.",
            ),
        ],
    )
    def test_rejects_invalid_arguments(
        self,
        query: str,
        candidate_top_k: int,
        message: str,
    ) -> None:
        """Tests that it rejects invalid arguments."""
        index = Mock()
        index.ntotal = 3

        with pytest.raises(ValueError, match=message):
            semantic_search(
                query=query,
                model=Mock(),
                index=index,
                candidate_top_k=candidate_top_k,
            )

    def test_rejects_empty_index(self) -> None:
        """Tests that it rejects an empty index."""
        index = Mock()
        index.ntotal = 0

        with pytest.raises(RuntimeError, match="The FAISS index is empty."):
            semantic_search(
                query="machine learning",
                model=Mock(),
                index=index,
                candidate_top_k=5,
            )

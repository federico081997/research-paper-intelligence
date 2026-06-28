"""Tests performed on the encoder module."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from research_paper_intelligence.embeddings.encoder import (
    generate_embeddings,
    get_model,
)

# -----------------------------------------------------------------------------
#   GetModel
# -----------------------------------------------------------------------------


class TestGetModel:
    """Tests the get_model function."""

    @patch(
        "research_paper_intelligence.embeddings.encoder.SentenceTransformer"
    )
    def test_get_model_returns_initialized_model(
        self,
        mock_sentence_transformer: MagicMock,
    ) -> None:
        """Test that the model is initialized correctly."""
        expected_model = MagicMock()
        mock_sentence_transformer.return_value = expected_model

        result = get_model("all-MiniLM-L6-v2")

        mock_sentence_transformer.assert_called_once_with("all-MiniLM-L6-v2")
        assert result is expected_model

    @patch(
        "research_paper_intelligence.embeddings.encoder.SentenceTransformer"
    )
    def test_get_model_propagates_loading_error(
        self,
        mock_sentence_transformer: MagicMock,
    ) -> None:
        """Test that loading errors are propagated."""
        mock_sentence_transformer.side_effect = OSError("Could not load model")

        with pytest.raises(OSError, match="Could not load model"):
            get_model("invalid-model")


# -----------------------------------------------------------------------------
#   GenerateEmbeddings
# -----------------------------------------------------------------------------


class TestGenerateEmbeddings:
    """Test the generate_embeddings function."""

    def test_generate_embeddings(self) -> None:
        """Tests that the embeddings are generated correctly."""
        model = MagicMock()
        model.encode.return_value = [
            [1.0, 2.0],
            [3.0, 4.0],
        ]

        result = generate_embeddings(
            model=model,
            texts=["First paper", "Second paper"],
            batch_size=16,
            device=torch.device("cpu"),
        )

        model.encode.assert_called_once_with(
            ["First paper", "Second paper"],
            batch_size=16,
            device="cpu",
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        expected = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float32

    def test_generate_embeddings_raises_for_empty_texts(self) -> None:
        """Tests that a ValueError is raised when no texts are provided."""
        model = MagicMock()

        with pytest.raises(
            ValueError,
            match="At least one text is required.",
        ):
            generate_embeddings(model=model, texts=[])

        model.encode.assert_not_called()

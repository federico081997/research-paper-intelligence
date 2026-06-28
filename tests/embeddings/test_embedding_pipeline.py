"""Tests performed of the embedding pipeline."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from research_paper_intelligence.config import Settings
from research_paper_intelligence.embeddings import embedding_pipeline
from research_paper_intelligence.embeddings.embedding_pipeline import (
    extract_embedding_texts,
)

# -----------------------------------------------------------------------------
#   TestExtractEmbeddingTexts
# -----------------------------------------------------------------------------


class TestExtractEmbeddingTexts:
    """Tests performed on the extract_embedding_texts function."""

    def test_extract_texts_as_list(
        self,
        sample_processed_dataframe: pd.DataFrame,
    ) -> None:
        """Test that the function returns a list of strings."""
        result = extract_embedding_texts(sample_processed_dataframe)

        assert isinstance(result, list)
        assert result == ["Paper A Summary A", "Paper B Summary B"]

    def test_raise_error_if_missing_column(
        self,
        sample_processed_dataframe: pd.DataFrame,
    ) -> None:
        """Tests that a missing required column raises a KeyError."""
        required_column = "combined_text"
        sample_processed_dataframe = sample_processed_dataframe.drop(
            columns=["combined_text"]
        )

        with pytest.raises(
            KeyError, match=f"Required column is missing: {required_column}"
        ):
            extract_embedding_texts(sample_processed_dataframe)

    @pytest.mark.parametrize("missing_value", [pd.NA, pd.NaT, np.nan, None])
    def test_raise_error_if_column_contains_missing_values(
        self,
        sample_processed_dataframe: pd.DataFrame,
        missing_value: Any,
    ) -> None:
        """Tests that missing values raise a ValueError."""
        sample_processed_dataframe["combined_text"] = [
            "Paper A Summary B",
            missing_value,
        ]

        with pytest.raises(
            ValueError,
            match="The combined_text column contains missing values.",
        ):
            extract_embedding_texts(sample_processed_dataframe)

    def test_raise_error_if_column_contains_empty(
        self,
        sample_processed_dataframe: pd.DataFrame,
    ) -> None:
        """Tests that empty values raise a ValueError."""
        column_name = "combined_text"

        sample_processed_dataframe[column_name] = [
            "Paper A Summary B",
            "",
        ]

        with pytest.raises(
            ValueError,
            match=f"The {column_name} column contains empty values.",
        ):
            extract_embedding_texts(sample_processed_dataframe)


# -----------------------------------------------------------------------------
#   TestRunEmbeddingPipeline
# -----------------------------------------------------------------------------


class TestRunEmbeddingPipeline:
    """Tests the run_embedding_pipeline function."""

    def test_run_embedding_pipeline_success(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_processed_dataframe: pd.DataFrame,
    ) -> None:
        """Tests that the run_embedding_pipeline function runs successfully."""
        # Set up the test data.
        processed_papers_path = tmp_path / "processed" / "processed-papers.csv"
        paper_embeddings_path = tmp_path / "processed" / "paper_embeddings.npy"
        sample_processed_dataframe = sample_processed_dataframe.copy()
        texts = ["Paper A Summary A", "Paper B Summary B"]
        device = "cpu"
        model = "model"
        embeddings = np.array([[1, 2, 3], [4, 5, 6]])

        # Set up the settings.
        settings = Settings(
            processed_papers_path=processed_papers_path,
            device="cpu",
            embedding_model_name="sentence_transformer_model",
            embedding_batch_size=16,
            paper_embeddings_path=paper_embeddings_path,
        )

        # Set up Mock functions.
        mock_load_data = Mock(return_value=sample_processed_dataframe)
        mock_extract_embedding_texts = Mock(return_value=texts)
        mock_get_device = Mock(return_value=device)
        mock_get_model = Mock(return_value=model)
        mock_generate_embeddings = Mock(return_value=embeddings)
        mock_save_embeddings = Mock()

        monkeypatch.setattr(embedding_pipeline, "load_data", mock_load_data)
        monkeypatch.setattr(
            embedding_pipeline,
            "extract_embedding_texts",
            mock_extract_embedding_texts,
        )
        monkeypatch.setattr(embedding_pipeline, "get_device", mock_get_device)
        monkeypatch.setattr(embedding_pipeline, "get_model", mock_get_model)
        monkeypatch.setattr(
            embedding_pipeline, "generate_embeddings", mock_generate_embeddings
        )
        monkeypatch.setattr(
            embedding_pipeline, "save_embeddings", mock_save_embeddings
        )

        result = embedding_pipeline.run_embedding_pipeline(settings)

        assert result == paper_embeddings_path

        # Check if each function is called once with appropriate parameters
        mock_load_data.assert_called_once_with(settings.processed_papers_path)
        mock_extract_embedding_texts.assert_called_once_with(
            sample_processed_dataframe
        )
        mock_get_device.assert_called_once_with(settings.device)
        mock_get_model.assert_called_once_with(settings.embedding_model_name)
        mock_generate_embeddings.assert_called_once_with(
            model=model,
            texts=texts,
            batch_size=settings.embedding_batch_size,
            device=device,
        )
        mock_save_embeddings.assert_called_once_with(
            embeddings=embeddings, path=settings.paper_embeddings_path
        )

        # Check that the embeddings are correct and that they have been saved
        # to the correct path
        actual_embeddings = mock_save_embeddings.call_args.kwargs["embeddings"]
        actual_path = mock_save_embeddings.call_args.kwargs["path"]

        np.testing.assert_array_equal(actual_embeddings, embeddings)
        assert actual_path == paper_embeddings_path

    def test_raises_error_if_embeddings_length_not_matching(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        sample_processed_dataframe: pd.DataFrame,
    ) -> None:
        """Tests if the embedding length does not match the number of texts."""
        # Set up the test data.
        processed_papers_path = tmp_path / "processed" / "processed-papers.csv"
        sample_processed_dataframe = sample_processed_dataframe.copy()
        texts = ["Paper A Summary A", "Paper B Summary B"]
        device = "cpu"
        model = "model"
        embeddings = np.array([[1, 2, 3]])

        # Set up the settings.
        settings = Settings(
            processed_papers_path=processed_papers_path,
            device="cpu",
            embedding_model_name="sentence_transformer_model",
            embedding_batch_size=16,
        )

        # Set up Mock functions.
        mock_load_data = Mock(return_value=sample_processed_dataframe)
        mock_extract_embedding_texts = Mock(return_value=texts)
        mock_get_device = Mock(return_value=device)
        mock_get_model = Mock(return_value=model)
        mock_generate_embeddings = Mock(return_value=embeddings)

        monkeypatch.setattr(embedding_pipeline, "load_data", mock_load_data)
        monkeypatch.setattr(
            embedding_pipeline,
            "extract_embedding_texts",
            mock_extract_embedding_texts,
        )
        monkeypatch.setattr(embedding_pipeline, "get_device", mock_get_device)
        monkeypatch.setattr(embedding_pipeline, "get_model", mock_get_model)
        monkeypatch.setattr(
            embedding_pipeline, "generate_embeddings", mock_generate_embeddings
        )

        with pytest.raises(
            RuntimeError,
            match="The number of generated embeddings does not match "
            "the number of papers.",
        ):
            embedding_pipeline.run_embedding_pipeline(settings)

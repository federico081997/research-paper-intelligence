"""Tests for the main generate embeddings script."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from research_paper_intelligence.cli import generate_embeddings as script


class TestGenerateEmbeddingsScript:
    """Tests for the embedding-generation entry point."""

    def test_uses_existing_embeddings(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that existing embeddings skip local generation."""
        settings = SimpleNamespace(
            hf_repository="test/repository",
            hf_paper_embeddings_file="embeddings.npy",
            paper_embeddings_path=Path("data/embeddings.npy"),
            hf_processed_papers_file="processed.csv",
            processed_papers_path=Path("data/processed.csv"),
        )

        mock_download = Mock(return_value=settings.paper_embeddings_path)
        mock_pipeline = Mock()

        monkeypatch.setattr(
            script, "get_settings", Mock(return_value=settings)
        )
        monkeypatch.setattr(script, "configure_logging", Mock())
        monkeypatch.setattr(script, "download_file", mock_download)
        monkeypatch.setattr(script, "run_embedding_pipeline", mock_pipeline)

        script.main()

        mock_download.assert_called_once_with(
            repository_id=settings.hf_repository,
            remote_filename=settings.hf_paper_embeddings_file,
            destination=settings.paper_embeddings_path,
            missing_ok=True,
        )
        mock_pipeline.assert_not_called()

    def test_generates_embeddings_when_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that missing embeddings trigger local generation."""
        settings = SimpleNamespace(
            hf_repository="test/repository",
            hf_paper_embeddings_file="embeddings.npy",
            paper_embeddings_path=Path("data/embeddings.npy"),
            hf_processed_papers_file="processed.csv",
            processed_papers_path=Path("data/processed.csv"),
        )

        mock_download = Mock(
            side_effect=[None, settings.processed_papers_path]
        )
        mock_pipeline = Mock()

        monkeypatch.setattr(
            script, "get_settings", Mock(return_value=settings)
        )
        monkeypatch.setattr(script, "configure_logging", Mock())
        monkeypatch.setattr(script, "download_file", mock_download)
        monkeypatch.setattr(script, "run_embedding_pipeline", mock_pipeline)

        script.main()

        assert mock_download.call_count == 2
        mock_download.assert_has_calls(
            [
                call(
                    repository_id=settings.hf_repository,
                    remote_filename=settings.hf_paper_embeddings_file,
                    destination=settings.paper_embeddings_path,
                    missing_ok=True,
                ),
                call(
                    repository_id=settings.hf_repository,
                    remote_filename=settings.hf_processed_papers_file,
                    destination=settings.processed_papers_path,
                ),
            ]
        )
        mock_pipeline.assert_called_once_with(settings)

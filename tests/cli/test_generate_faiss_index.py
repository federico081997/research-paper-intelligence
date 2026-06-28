"""Tests for the FAISS-index preparation script."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

import numpy as np
import pytest

from research_paper_intelligence.cli import (
    generate_faiss_index as script,
)


class TestGenerateFaissIndexScript:
    """Tests for the FAISS-index preparation script."""

    def test_uses_existing_faiss_index(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that local generation is skipped when an index is available."""
        settings = SimpleNamespace(
            hf_repository="test/repository",
            hf_faiss_index_papers_file="indexes/papers.faiss",
            faiss_index_papers_path=Path("data/papers.faiss"),
            hf_paper_embeddings_file="embeddings/papers.npy",
            paper_embeddings_path=Path("data/papers.npy"),
        )

        mock_download = Mock(return_value=settings.faiss_index_papers_path)
        mock_load = Mock()
        mock_build = Mock()
        mock_save = Mock()

        monkeypatch.setattr(
            script,
            "get_settings",
            Mock(return_value=settings),
        )
        monkeypatch.setattr(script, "configure_logging", Mock())
        monkeypatch.setattr(script, "download_file", mock_download)
        monkeypatch.setattr(script, "load_embeddings", mock_load)
        monkeypatch.setattr(script, "build_faiss_index", mock_build)
        monkeypatch.setattr(script, "save_faiss_index", mock_save)

        script.main()

        mock_download.assert_called_once_with(
            repository_id=settings.hf_repository,
            remote_filename=settings.hf_faiss_index_papers_file,
            destination=settings.faiss_index_papers_path,
            missing_ok=True,
        )
        mock_load.assert_not_called()
        mock_build.assert_not_called()
        mock_save.assert_not_called()

    def test_builds_faiss_index_when_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that a missing index is built from paper embeddings."""
        settings = SimpleNamespace(
            hf_repository="test/repository",
            hf_faiss_index_papers_file="indexes/papers.faiss",
            faiss_index_papers_path=Path("data/papers.faiss"),
            hf_paper_embeddings_file="embeddings/papers.npy",
            paper_embeddings_path=Path("data/papers.npy"),
        )

        embeddings = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float32,
        )
        index = SimpleNamespace(ntotal=2)

        mock_download = Mock(
            side_effect=[None, settings.paper_embeddings_path]
        )
        mock_load = Mock(return_value=embeddings)
        mock_build = Mock(return_value=index)
        mock_save = Mock()

        monkeypatch.setattr(
            script,
            "get_settings",
            Mock(return_value=settings),
        )
        monkeypatch.setattr(script, "configure_logging", Mock())
        monkeypatch.setattr(script, "download_file", mock_download)
        monkeypatch.setattr(script, "load_embeddings", mock_load)
        monkeypatch.setattr(script, "build_faiss_index", mock_build)
        monkeypatch.setattr(script, "save_faiss_index", mock_save)

        script.main()

        mock_download.assert_has_calls(
            [
                call(
                    repository_id=settings.hf_repository,
                    remote_filename=settings.hf_faiss_index_papers_file,
                    destination=settings.faiss_index_papers_path,
                    missing_ok=True,
                ),
                call(
                    repository_id=settings.hf_repository,
                    remote_filename=settings.hf_paper_embeddings_file,
                    destination=settings.paper_embeddings_path,
                ),
            ]
        )

        mock_load.assert_called_once_with(settings.paper_embeddings_path)
        mock_build.assert_called_once()
        assert mock_build.call_args.args[0] is embeddings

        mock_save.assert_called_once_with(
            index=index,
            output_path=settings.faiss_index_papers_path,
        )

"""Tests for the TF-IDF artifact preparation main entry point."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

import research_paper_intelligence.cli.generate_tfidf_index as script


class TestGenerateTfidfIndex:
    """Tests for the TF-IDF preparation entry point."""

    def test_uses_existing_tfidf_artifacts(
        self, monkeypatch: pytest.MonkeyPatch, simple_settings: SimpleNamespace
    ) -> None:
        """Existing artifacts should be downloaded without rebuilding them."""
        mock_download = Mock(
            side_effect=[
                simple_settings.tfidf_vectorizer_path,
                simple_settings.tfidf_matrix_path,
            ]
        )

        monkeypatch.setattr(
            script, "get_settings", Mock(return_value=simple_settings)
        )
        monkeypatch.setattr(script, "configure_logging", Mock())
        monkeypatch.setattr(script, "download_file", mock_download)

        script.main()

        assert mock_download.call_count == 2

    def test_builds_tfidf_artifacts_when_missing(
        self,
        monkeypatch: pytest.MonkeyPatch,
        simple_settings: SimpleNamespace,
    ) -> None:
        """Missing artifacts should be built and saved locally."""
        dataframe = MagicMock()
        papers = [MagicMock(), MagicMock()]
        texts = ["First paper", "Second paper"]

        vectorizer = MagicMock()
        matrix = MagicMock()
        matrix.shape = (2, 10)

        repository = MagicMock()
        repository.get_all.return_value = papers
        repository_class = Mock(return_value=repository)

        # Mock necessary functions.
        mock_download = Mock(
            side_effect=[
                None,
                simple_settings.tfidf_matrix_path,
                simple_settings.processed_papers_path,
            ]
        )
        mock_build = Mock(return_value=(vectorizer, matrix))
        mock_load_data = Mock(return_value=dataframe)
        mock_corpus = Mock(return_value=texts)
        mock_save = Mock()

        monkeypatch.setattr(
            script, "get_settings", Mock(return_value=simple_settings)
        )
        monkeypatch.setattr(script, "configure_logging", Mock())
        monkeypatch.setattr(script, "download_file", mock_download)
        monkeypatch.setattr(script, "load_data", mock_load_data)
        monkeypatch.setattr(
            script,
            "PaperRepository",
            repository_class,
        )
        monkeypatch.setattr(
            script,
            "create_lexical_corpus",
            mock_corpus,
        )
        monkeypatch.setattr(script, "build_tfidf_index", mock_build)
        monkeypatch.setattr(script, "save_tfidf_index", mock_save)

        script.main()

        assert mock_download.call_count == 3

        mock_load_data.assert_called_once_with(
            simple_settings.processed_papers_path
        )
        repository_class.assert_called_once_with(dataframe)
        repository.get_all.assert_called_once_with()
        mock_corpus.assert_called_once_with(papers)
        mock_build.assert_called_once_with(texts)

        mock_save.assert_called_once_with(
            vectorizer=vectorizer,
            matrix=matrix,
            vectorizer_path=simple_settings.tfidf_vectorizer_path,
            matrix_path=simple_settings.tfidf_matrix_path,
        )

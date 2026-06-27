"""Tests for the preprocessing CLI entry point."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from research_paper_intelligence.cli import preprocess_dataset as cli
from research_paper_intelligence.config import Settings


class TestPreprocessDataset:
    """Tests for the preprocessing CLI entry point."""

    @staticmethod
    def test_main_runs_preprocessing_workflow(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Configure, download, preprocess, and log completion."""
        settings = Settings.model_construct(
            hf_repository="user/repository",
            hf_raw_papers_file="data/raw/arxiv_papers.csv",
            raw_papers_path=Path("data/raw/arxiv_papers.csv"),
        )

        mock_get_settings = Mock(return_value=settings)
        mock_configure_logging = Mock()
        mock_download_file = Mock()
        mock_run_pipeline = Mock()
        mock_logger = Mock()
        mock_perf_counter = Mock(side_effect=[10.0, 12.5])

        monkeypatch.setattr(cli, "get_settings", mock_get_settings)
        monkeypatch.setattr(cli, "configure_logging", mock_configure_logging)
        monkeypatch.setattr(cli, "download_file", mock_download_file)
        monkeypatch.setattr(
            cli,
            "run_preprocessing_pipeline",
            mock_run_pipeline,
        )
        monkeypatch.setattr(cli, "logger", mock_logger)
        monkeypatch.setattr(cli, "perf_counter", mock_perf_counter)

        cli.main()

        mock_get_settings.assert_called_once_with()
        mock_configure_logging.assert_called_once_with(settings)
        mock_download_file.assert_called_once_with(
            settings.hf_repository,
            settings.hf_raw_papers_file,
            settings.raw_papers_path,
        )
        mock_run_pipeline.assert_called_once_with(settings)
        mock_logger.info.assert_called_once_with(
            "Preprocessing pipeline completed successfully in %.2f seconds",
            2.5,
        )

"""Tests for the preprocessing CLI entry point."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from research_paper_intelligence.cli import preprocess_dataset as script

# -----------------------------------------------------------------------------
#   TestPreprocessDataset
# -----------------------------------------------------------------------------


class TestPreprocessDatasetScript:
    """Tests for the preprocessing entry point."""

    def test_main_downloads_and_preprocesses(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that main downloads raw papers and runs preprocessing."""
        settings = SimpleNamespace(
            hf_repository="test/repository",
            hf_raw_papers_file="raw/papers.csv",
            raw_papers_path=Path("data/raw/papers.csv"),
        )

        mock_get_settings = Mock(return_value=settings)
        mock_configure_logging = Mock()
        mock_download_file = Mock()
        mock_run_pipeline = Mock()
        mock_perf_counter = Mock(side_effect=[10.0, 12.5])
        mock_logger = Mock()

        monkeypatch.setattr(script, "get_settings", mock_get_settings)
        monkeypatch.setattr(
            script,
            "configure_logging",
            mock_configure_logging,
        )
        monkeypatch.setattr(script, "download_file", mock_download_file)
        monkeypatch.setattr(
            script,
            "run_preprocessing_pipeline",
            mock_run_pipeline,
        )
        monkeypatch.setattr(script, "perf_counter", mock_perf_counter)
        monkeypatch.setattr(script, "logger", mock_logger)

        script.main()

        mock_get_settings.assert_called_once_with()
        mock_configure_logging.assert_called_once_with(settings)
        mock_download_file.assert_called_once_with(
            repository_id="test/repository",
            remote_filename="raw/papers.csv",
            destination=Path("data/raw/papers.csv"),
        )
        mock_run_pipeline.assert_called_once_with(settings)
        mock_logger.info.assert_called_once_with(
            "Preprocessing pipeline completed successfully in %.2f seconds.",
            2.5,
        )

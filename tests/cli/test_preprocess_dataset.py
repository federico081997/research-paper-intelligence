"""Tests performed on the preprocess_dataset script."""

from unittest.mock import Mock

import pytest

from research_paper_intelligence.cli import preprocess_dataset as cli
from research_paper_intelligence.config import Settings

# -----------------------------------------------------------------------------
#   PreprocessDataset
# -----------------------------------------------------------------------------


class TestPreprocessDataset:
    """Tests performed on the ``preprocess_dataset`` script."""

    @staticmethod
    def test_main_configures_and_runs_pipeline(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Load settings, configure logging, and run preprocessing."""
        settings = Settings.model_construct()

        mock_get_settings = Mock(return_value=settings)
        mock_configure_logging = Mock()
        mock_run_pipeline = Mock()

        monkeypatch.setattr(cli, "get_settings", mock_get_settings)
        monkeypatch.setattr(cli, "configure_logging", mock_configure_logging)
        monkeypatch.setattr(
            cli,
            "run_preprocessing_pipeline",
            mock_run_pipeline,
        )

        cli.main()

        mock_get_settings.assert_called_once_with()
        mock_configure_logging.assert_called_once_with(settings)
        mock_run_pipeline.assert_called_once_with(settings)

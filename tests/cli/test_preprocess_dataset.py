"""Tests for the research-paper preprocessing CLI."""

import logging
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from research_paper_intelligence.cli import preprocess_dataset
from research_paper_intelligence.cli.preprocess_dataset import main
from research_paper_intelligence.config import Settings


@pytest.fixture
def preprocessing_settings() -> Mock:
    """Create settings required by the preprocessing CLI."""
    settings = Mock(spec=Settings)
    settings.hf_repository = "user/research-papers"
    settings.hf_raw_papers_file = "data/raw/raw_papers.csv"
    settings.raw_papers_path = Path("data/raw/raw_papers.csv")
    settings.processed_papers_path = Path(
        "data/processed/processed_papers.csv"
    )

    return settings


class TestPreprocessDatasetMain:
    """Tests for the dataset-preprocessing CLI entry point."""

    def test_downloads_raw_data_and_runs_preprocessing_pipeline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        preprocessing_settings: Mock,
    ) -> None:
        """Download the raw dataset and run the preprocessing pipeline."""
        get_settings_mock = Mock(return_value=preprocessing_settings)
        configure_logging_mock = Mock()
        download_file_mock = Mock(
            return_value=preprocessing_settings.raw_papers_path
        )
        run_preprocessing_pipeline_mock = Mock()

        monkeypatch.setattr(
            preprocess_dataset,
            "get_settings",
            get_settings_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "configure_logging",
            configure_logging_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "run_preprocessing_pipeline",
            run_preprocessing_pipeline_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "perf_counter",
            Mock(side_effect=[10.0, 12.0]),
        )

        main()

        get_settings_mock.assert_called_once_with()
        configure_logging_mock.assert_called_once_with(preprocessing_settings)
        download_file_mock.assert_called_once_with(
            repository_id=preprocessing_settings.hf_repository,
            remote_filename=preprocessing_settings.hf_raw_papers_file,
            destination=preprocessing_settings.raw_papers_path,
        )
        run_preprocessing_pipeline_mock.assert_called_once_with(
            preprocessing_settings
        )

    def test_executes_preprocessing_steps_in_expected_order(
        self,
        monkeypatch: pytest.MonkeyPatch,
        preprocessing_settings: Mock,
    ) -> None:
        """Configure logging, download data, and then run preprocessing."""
        workflow = Mock()

        configure_logging_mock = Mock()
        download_file_mock = Mock()
        run_preprocessing_pipeline_mock = Mock()

        workflow.attach_mock(
            configure_logging_mock,
            "configure_logging",
        )
        workflow.attach_mock(
            download_file_mock,
            "download_file",
        )
        workflow.attach_mock(
            run_preprocessing_pipeline_mock,
            "run_preprocessing_pipeline",
        )

        monkeypatch.setattr(
            preprocess_dataset,
            "get_settings",
            Mock(return_value=preprocessing_settings),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "configure_logging",
            configure_logging_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "run_preprocessing_pipeline",
            run_preprocessing_pipeline_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "perf_counter",
            Mock(side_effect=[1.0, 2.0]),
        )

        main()

        assert workflow.mock_calls == [
            call.configure_logging(preprocessing_settings),
            call.download_file(
                repository_id=preprocessing_settings.hf_repository,
                remote_filename=(preprocessing_settings.hf_raw_papers_file),
                destination=preprocessing_settings.raw_papers_path,
            ),
            call.run_preprocessing_pipeline(preprocessing_settings),
        ]

    def test_logs_preprocessing_workflow(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        preprocessing_settings: Mock,
    ) -> None:
        """Log the preprocessing input and output paths."""
        monkeypatch.setattr(
            preprocess_dataset,
            "get_settings",
            Mock(return_value=preprocessing_settings),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "download_file",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "run_preprocessing_pipeline",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "perf_counter",
            Mock(side_effect=[10.0, 12.0]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=preprocess_dataset.__name__,
        ):
            main()

        assert "Preparing processed research-paper data." in caplog.messages
        assert (
            "Preprocessing raw papers from "
            f"{preprocessing_settings.raw_papers_path}." in caplog.messages
        )
        assert (
            "Processed papers were saved to "
            f"{preprocessing_settings.processed_papers_path}."
            in caplog.messages
        )

    def test_logs_total_elapsed_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        preprocessing_settings: Mock,
    ) -> None:
        """Log the total preprocessing time with two decimal places."""
        monkeypatch.setattr(
            preprocess_dataset,
            "get_settings",
            Mock(return_value=preprocessing_settings),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "download_file",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "run_preprocessing_pipeline",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "perf_counter",
            Mock(side_effect=[100.0, 103.456]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=preprocess_dataset.__name__,
        ):
            main()

        assert (
            "Research-paper preprocessing completed successfully "
            "in 3.46 seconds." in caplog.messages
        )

    def test_propagates_raw_data_download_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        preprocessing_settings: Mock,
    ) -> None:
        """Propagate errors raised while downloading the raw dataset."""
        run_preprocessing_pipeline_mock = Mock()

        monkeypatch.setattr(
            preprocess_dataset,
            "get_settings",
            Mock(return_value=preprocessing_settings),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "download_file",
            Mock(side_effect=RuntimeError("Raw dataset download failed.")),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "run_preprocessing_pipeline",
            run_preprocessing_pipeline_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="Raw dataset download failed",
        ):
            main()

        run_preprocessing_pipeline_mock.assert_not_called()

    def test_propagates_preprocessing_pipeline_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        preprocessing_settings: Mock,
    ) -> None:
        """Propagate errors raised by the preprocessing pipeline."""
        download_file_mock = Mock()

        monkeypatch.setattr(
            preprocess_dataset,
            "get_settings",
            Mock(return_value=preprocessing_settings),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "run_preprocessing_pipeline",
            Mock(side_effect=RuntimeError("Dataset preprocessing failed.")),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="Dataset preprocessing failed",
        ):
            main()

        download_file_mock.assert_called_once_with(
            repository_id=preprocessing_settings.hf_repository,
            remote_filename=preprocessing_settings.hf_raw_papers_file,
            destination=preprocessing_settings.raw_papers_path,
        )

    def test_does_not_log_success_when_preprocessing_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        preprocessing_settings: Mock,
    ) -> None:
        """Avoid logging successful completion after a pipeline failure."""
        monkeypatch.setattr(
            preprocess_dataset,
            "get_settings",
            Mock(return_value=preprocessing_settings),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "download_file",
            Mock(),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "run_preprocessing_pipeline",
            Mock(side_effect=RuntimeError("Dataset preprocessing failed.")),
        )
        monkeypatch.setattr(
            preprocess_dataset,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with caplog.at_level(
            logging.INFO,
            logger=preprocess_dataset.__name__,
        ):
            with pytest.raises(RuntimeError):
                main()

        assert not any(
            message.startswith("Processed papers were saved to")
            for message in caplog.messages
        )
        assert not any(
            message.startswith(
                "Research-paper preprocessing completed successfully"
            )
            for message in caplog.messages
        )

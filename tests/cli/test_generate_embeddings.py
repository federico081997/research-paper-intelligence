"""Tests for the embedding-generation CLI."""

import logging
from pathlib import Path
from unittest.mock import Mock, call

import pytest

from research_paper_intelligence.cli import generate_embeddings
from research_paper_intelligence.cli.generate_embeddings import main
from research_paper_intelligence.config import Settings


@pytest.fixture
def embedding_settings() -> Mock:
    """Create settings required by the embedding-generation CLI."""
    settings = Mock(spec=Settings)
    settings.hf_repository = "user/research-papers"
    settings.hf_paper_embeddings_file = "artifacts/paper_embeddings.npy"
    settings.paper_embeddings_path = Path(
        "data/artifacts/paper_embeddings.npy"
    )
    settings.hf_processed_papers_file = "data/processed/processed_papers.csv"
    settings.processed_papers_path = Path(
        "data/processed/processed_papers.csv"
    )

    return settings


class TestGenerateEmbeddingsMain:
    """Tests for the embedding-generation CLI entry point."""

    def test_uses_existing_embeddings_when_download_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        embedding_settings: Mock,
    ) -> None:
        """Use downloaded embeddings without running the local pipeline."""
        embeddings_path = Path("data/artifacts/paper_embeddings.npy")
        get_settings_mock = Mock(return_value=embedding_settings)
        configure_logging_mock = Mock()
        download_file_mock = Mock(return_value=embeddings_path)
        run_embedding_pipeline_mock = Mock()
        perf_counter_mock = Mock(side_effect=[10.0, 12.5])

        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            get_settings_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            configure_logging_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            run_embedding_pipeline_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            perf_counter_mock,
        )

        main()

        get_settings_mock.assert_called_once_with()
        configure_logging_mock.assert_called_once_with(embedding_settings)
        download_file_mock.assert_called_once_with(
            repository_id=embedding_settings.hf_repository,
            remote_filename=(embedding_settings.hf_paper_embeddings_file),
            destination=embedding_settings.paper_embeddings_path,
            missing_ok=True,
        )
        run_embedding_pipeline_mock.assert_not_called()
        assert perf_counter_mock.call_count == 2

    def test_generates_embeddings_when_download_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        embedding_settings: Mock,
    ) -> None:
        """Generate embeddings locally when no precomputed file exists."""
        get_settings_mock = Mock(return_value=embedding_settings)
        configure_logging_mock = Mock()
        download_file_mock = Mock(
            side_effect=[
                None,
                embedding_settings.processed_papers_path,
            ]
        )
        run_embedding_pipeline_mock = Mock()
        perf_counter_mock = Mock(side_effect=[20.0, 24.25])

        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            get_settings_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            configure_logging_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            run_embedding_pipeline_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            perf_counter_mock,
        )

        main()

        assert download_file_mock.call_args_list == [
            call(
                repository_id=embedding_settings.hf_repository,
                remote_filename=(embedding_settings.hf_paper_embeddings_file),
                destination=(embedding_settings.paper_embeddings_path),
                missing_ok=True,
            ),
            call(
                repository_id=embedding_settings.hf_repository,
                remote_filename=(embedding_settings.hf_processed_papers_file),
                destination=embedding_settings.processed_papers_path,
            ),
        ]
        run_embedding_pipeline_mock.assert_called_once_with(embedding_settings)

    def test_configures_logging_with_application_settings(
        self,
        monkeypatch: pytest.MonkeyPatch,
        embedding_settings: Mock,
    ) -> None:
        """Configure logging using the loaded application settings."""
        configure_logging_mock = Mock()

        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            Mock(return_value=embedding_settings),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            configure_logging_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            Mock(return_value=Path("data/artifacts/paper_embeddings.npy")),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            Mock(side_effect=[1.0, 2.0]),
        )

        main()

        configure_logging_mock.assert_called_once_with(embedding_settings)

    def test_logs_existing_embeddings_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        embedding_settings: Mock,
    ) -> None:
        """Log the path used for existing paper embeddings."""
        embeddings_path = Path("data/artifacts/paper_embeddings.npy")

        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            Mock(return_value=embedding_settings),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            Mock(return_value=embeddings_path),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            Mock(side_effect=[10.0, 12.0]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_embeddings.__name__,
        ):
            main()

        assert "Preparing paper embeddings." in caplog.messages
        assert (
            "Using the existing paper embeddings at "
            f"{embeddings_path}." in caplog.messages
        )

    def test_logs_local_embedding_generation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        embedding_settings: Mock,
    ) -> None:
        """Log the local embedding-generation workflow."""
        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            Mock(return_value=embedding_settings),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    embedding_settings.processed_papers_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            Mock(side_effect=[5.0, 8.0]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_embeddings.__name__,
        ):
            main()

        assert (
            "Precomputed paper embeddings are unavailable. "
            "Generating them locally from the processed papers."
            in caplog.messages
        )
        assert (
            "Generating paper embeddings from "
            f"{embedding_settings.processed_papers_path}." in caplog.messages
        )
        assert (
            "Paper embeddings were generated and saved to "
            f"{embedding_settings.paper_embeddings_path}." in caplog.messages
        )

    def test_logs_total_elapsed_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        embedding_settings: Mock,
    ) -> None:
        """Log the total preparation time with two decimal places."""
        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            Mock(return_value=embedding_settings),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            Mock(return_value=Path("data/artifacts/paper_embeddings.npy")),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            Mock(side_effect=[100.0, 103.456]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_embeddings.__name__,
        ):
            main()

        assert (
            "Paper embedding preparation completed successfully "
            "in 3.46 seconds." in caplog.messages
        )

    def test_does_not_generate_embeddings_when_existing_file_is_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        embedding_settings: Mock,
    ) -> None:
        """Skip downloading data and running the pipeline when cached."""
        download_file_mock = Mock(
            return_value=Path("data/artifacts/paper_embeddings.npy")
        )
        run_embedding_pipeline_mock = Mock()

        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            Mock(return_value=embedding_settings),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            run_embedding_pipeline_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            Mock(side_effect=[1.0, 2.0]),
        )

        main()

        assert download_file_mock.call_count == 1
        run_embedding_pipeline_mock.assert_not_called()

    def test_propagates_processed_data_download_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        embedding_settings: Mock,
    ) -> None:
        """Propagate errors raised while downloading processed papers."""
        download_file_mock = Mock(
            side_effect=[
                None,
                RuntimeError("Processed-paper download failed."),
            ]
        )
        run_embedding_pipeline_mock = Mock()

        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            Mock(return_value=embedding_settings),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            run_embedding_pipeline_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="Processed-paper download failed",
        ):
            main()

        run_embedding_pipeline_mock.assert_not_called()

    def test_propagates_embedding_pipeline_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        embedding_settings: Mock,
    ) -> None:
        """Propagate errors raised by the local embedding pipeline."""
        run_embedding_pipeline_mock = Mock(
            side_effect=RuntimeError("Embedding generation failed.")
        )

        monkeypatch.setattr(
            generate_embeddings,
            "get_settings",
            Mock(return_value=embedding_settings),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    embedding_settings.processed_papers_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_embeddings,
            "run_embedding_pipeline",
            run_embedding_pipeline_mock,
        )
        monkeypatch.setattr(
            generate_embeddings,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="Embedding generation failed",
        ):
            main()

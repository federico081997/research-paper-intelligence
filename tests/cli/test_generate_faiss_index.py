"""Tests for the FAISS-index generation CLI."""

import logging
from pathlib import Path
from unittest.mock import Mock, call

import numpy as np
import pytest

from research_paper_intelligence.cli import generate_faiss_index
from research_paper_intelligence.cli.generate_faiss_index import main
from research_paper_intelligence.config import Settings


@pytest.fixture
def faiss_settings() -> Mock:
    """Create settings required by the FAISS-index CLI."""
    settings = Mock(spec=Settings)
    settings.hf_repository = "user/research-papers"
    settings.hf_faiss_index_papers_file = "artifacts/faiss_index_papers.bin"
    settings.faiss_index_papers_path = Path(
        "data/artifacts/faiss_index_papers.bin"
    )
    settings.hf_paper_embeddings_file = "artifacts/paper_embeddings.npy"
    settings.paper_embeddings_path = Path(
        "data/artifacts/paper_embeddings.npy"
    )

    return settings


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Create representative paper embeddings."""
    return np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )


class TestGenerateFaissIndexMain:
    """Tests for the FAISS-index CLI entry point."""

    def test_uses_existing_index_when_download_succeeds(
        self,
        monkeypatch: pytest.MonkeyPatch,
        faiss_settings: Mock,
    ) -> None:
        """Use the downloaded index without building a local index."""
        existing_index_path = Path("data/artifacts/faiss_index_papers.bin")
        get_settings_mock = Mock(return_value=faiss_settings)
        configure_logging_mock = Mock()
        download_file_mock = Mock(return_value=existing_index_path)
        load_embeddings_mock = Mock()
        build_faiss_index_mock = Mock()
        save_faiss_index_mock = Mock()

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            get_settings_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            configure_logging_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            load_embeddings_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            build_faiss_index_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            save_faiss_index_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(side_effect=[10.0, 12.5]),
        )

        main()

        get_settings_mock.assert_called_once_with()
        configure_logging_mock.assert_called_once_with(faiss_settings)
        download_file_mock.assert_called_once_with(
            repository_id=faiss_settings.hf_repository,
            remote_filename=(faiss_settings.hf_faiss_index_papers_file),
            destination=faiss_settings.faiss_index_papers_path,
            missing_ok=True,
        )
        load_embeddings_mock.assert_not_called()
        build_faiss_index_mock.assert_not_called()
        save_faiss_index_mock.assert_not_called()

    def test_builds_index_when_precomputed_index_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        faiss_settings: Mock,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Build and save an index when no precomputed index exists."""
        index = Mock()
        index.ntotal = sample_embeddings.shape[0]

        download_file_mock = Mock(
            side_effect=[
                None,
                faiss_settings.paper_embeddings_path,
            ]
        )
        load_embeddings_mock = Mock(return_value=sample_embeddings)
        build_faiss_index_mock = Mock(return_value=index)
        save_faiss_index_mock = Mock()

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            load_embeddings_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            build_faiss_index_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            save_faiss_index_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(side_effect=[20.0, 24.0]),
        )

        main()

        assert download_file_mock.call_args_list == [
            call(
                repository_id=faiss_settings.hf_repository,
                remote_filename=(faiss_settings.hf_faiss_index_papers_file),
                destination=faiss_settings.faiss_index_papers_path,
                missing_ok=True,
            ),
            call(
                repository_id=faiss_settings.hf_repository,
                remote_filename=(faiss_settings.hf_paper_embeddings_file),
                destination=faiss_settings.paper_embeddings_path,
            ),
        ]
        load_embeddings_mock.assert_called_once_with(
            faiss_settings.paper_embeddings_path
        )
        build_faiss_index_mock.assert_called_once_with(sample_embeddings)
        save_faiss_index_mock.assert_called_once_with(
            index=index,
            output_path=faiss_settings.faiss_index_papers_path,
        )

    def test_passes_loaded_embeddings_to_index_builder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        faiss_settings: Mock,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Pass the exact loaded embedding array to the index builder."""
        index = Mock()
        index.ntotal = sample_embeddings.shape[0]
        build_faiss_index_mock = Mock(return_value=index)

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    faiss_settings.paper_embeddings_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            Mock(return_value=sample_embeddings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            build_faiss_index_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(side_effect=[1.0, 2.0]),
        )

        main()

        built_embeddings = build_faiss_index_mock.call_args.args[0]

        assert built_embeddings is sample_embeddings

    def test_saves_built_index_to_configured_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        faiss_settings: Mock,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Save the locally built index to the configured path."""
        index = Mock()
        index.ntotal = sample_embeddings.shape[0]
        save_faiss_index_mock = Mock()

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    faiss_settings.paper_embeddings_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            Mock(return_value=sample_embeddings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            Mock(return_value=index),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            save_faiss_index_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(side_effect=[1.0, 2.0]),
        )

        main()

        save_faiss_index_mock.assert_called_once_with(
            index=index,
            output_path=faiss_settings.faiss_index_papers_path,
        )

    def test_logs_existing_index_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        faiss_settings: Mock,
    ) -> None:
        """Log the location of an existing downloaded index."""
        existing_index_path = Path("data/artifacts/faiss_index_papers.bin")

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            Mock(return_value=existing_index_path),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(side_effect=[10.0, 12.0]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_faiss_index.__name__,
        ):
            main()

        assert "Preparing the FAISS index." in caplog.messages
        assert (
            f"Using the existing FAISS index at {existing_index_path}."
            in caplog.messages
        )

    def test_logs_local_index_build_information(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        faiss_settings: Mock,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Log the local index-building workflow and vector count."""
        index = Mock()
        index.ntotal = sample_embeddings.shape[0]

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    faiss_settings.paper_embeddings_path,
                ],
            ),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            Mock(return_value=sample_embeddings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            Mock(return_value=index),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(side_effect=[10.0, 13.0]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_faiss_index.__name__,
        ):
            main()

        assert (
            "A precomputed FAISS index is unavailable. "
            "Building the index locally from paper embeddings."
            in caplog.messages
        )
        assert (
            f"Building a FAISS index from {sample_embeddings.shape[0]} "
            "embeddings." in caplog.messages
        )
        assert (
            "Saved the FAISS index containing "
            f"{index.ntotal} vectors to "
            f"{faiss_settings.faiss_index_papers_path}." in caplog.messages
        )

    def test_logs_total_elapsed_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        faiss_settings: Mock,
    ) -> None:
        """Log total preparation time with two decimal places."""
        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            Mock(return_value=Path("data/artifacts/faiss_index_papers.bin")),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(side_effect=[100.0, 103.456]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_faiss_index.__name__,
        ):
            main()

        assert (
            "FAISS index preparation completed successfully "
            "in 3.46 seconds." in caplog.messages
        )

    def test_propagates_embedding_download_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        faiss_settings: Mock,
    ) -> None:
        """Propagate errors raised while downloading embeddings."""
        download_file_mock = Mock(
            side_effect=[
                None,
                RuntimeError("Embedding download failed."),
            ]
        )
        load_embeddings_mock = Mock()

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            load_embeddings_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="Embedding download failed",
        ):
            main()

        load_embeddings_mock.assert_not_called()

    def test_propagates_embedding_loading_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        faiss_settings: Mock,
    ) -> None:
        """Propagate errors raised while loading embeddings."""
        build_faiss_index_mock = Mock()

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    faiss_settings.paper_embeddings_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            Mock(side_effect=RuntimeError("Embedding loading failed.")),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            build_faiss_index_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="Embedding loading failed",
        ):
            main()

        build_faiss_index_mock.assert_not_called()

    def test_propagates_index_building_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        faiss_settings: Mock,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Propagate errors raised while building the FAISS index."""
        save_faiss_index_mock = Mock()

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    faiss_settings.paper_embeddings_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            Mock(return_value=sample_embeddings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            Mock(side_effect=RuntimeError("FAISS index building failed.")),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            save_faiss_index_mock,
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="FAISS index building failed",
        ):
            main()

        save_faiss_index_mock.assert_not_called()

    def test_propagates_index_saving_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        faiss_settings: Mock,
        sample_embeddings: np.ndarray,
    ) -> None:
        """Propagate errors raised while saving the FAISS index."""
        index = Mock()
        index.ntotal = sample_embeddings.shape[0]

        monkeypatch.setattr(
            generate_faiss_index,
            "get_settings",
            Mock(return_value=faiss_settings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    faiss_settings.paper_embeddings_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "load_embeddings",
            Mock(return_value=sample_embeddings),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "build_faiss_index",
            Mock(return_value=index),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "save_faiss_index",
            Mock(side_effect=RuntimeError("FAISS index saving failed.")),
        )
        monkeypatch.setattr(
            generate_faiss_index,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="FAISS index saving failed",
        ):
            main()

"""Tests for the TF-IDF index generation CLI."""

import logging
from pathlib import Path
from unittest.mock import Mock, call

import pytest
from scipy.sparse import csr_matrix

from research_paper_intelligence.cli import generate_tfidf_index
from research_paper_intelligence.cli.generate_tfidf_index import main


@pytest.fixture
def tfidf_settings() -> Mock:
    """Create settings required by the TF-IDF generation CLI."""
    settings = Mock()
    settings.hf_repository = "user/research-papers"
    settings.hf_tfidf_vectorizer_file = "artifacts/tfidf_vectorizer.joblib"
    settings.hf_tfidf_matrix_file = "artifacts/tfidf_matrix.npz"
    settings.hf_processed_papers_file = "data/processed/processed_papers.csv"
    settings.tfidf_vectorizer_path = Path(
        "data/artifacts/tfidf_vectorizer.joblib"
    )
    settings.tfidf_matrix_path = Path("data/artifacts/tfidf_matrix.npz")
    settings.processed_papers_path = Path(
        "data/processed/processed_papers.csv"
    )

    return settings


class TestGenerateTfidfIndexMain:
    """Tests for the TF-IDF index generation CLI entry point."""

    def test_uses_existing_artifacts_when_both_downloads_succeed(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tfidf_settings: Mock,
    ) -> None:
        """Use downloaded artifacts without building them locally."""
        vectorizer_path = tfidf_settings.tfidf_vectorizer_path
        matrix_path = tfidf_settings.tfidf_matrix_path

        get_settings_mock = Mock(return_value=tfidf_settings)
        configure_logging_mock = Mock()
        download_file_mock = Mock(
            side_effect=[
                vectorizer_path,
                matrix_path,
            ]
        )
        load_data_mock = Mock()
        repository_mock = Mock()
        create_lexical_corpus_mock = Mock()
        build_tfidf_index_mock = Mock()
        save_tfidf_index_mock = Mock()

        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            get_settings_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            configure_logging_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "load_data",
            load_data_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "PaperRepository",
            repository_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "create_lexical_corpus",
            create_lexical_corpus_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "build_tfidf_index",
            build_tfidf_index_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "save_tfidf_index",
            save_tfidf_index_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(side_effect=[10.0, 12.0]),
        )

        main()

        get_settings_mock.assert_called_once_with()
        configure_logging_mock.assert_called_once_with(tfidf_settings)

        assert download_file_mock.call_args_list == [
            call(
                repository_id=tfidf_settings.hf_repository,
                remote_filename=(tfidf_settings.hf_tfidf_vectorizer_file),
                destination=tfidf_settings.tfidf_vectorizer_path,
                missing_ok=True,
            ),
            call(
                repository_id=tfidf_settings.hf_repository,
                remote_filename=tfidf_settings.hf_tfidf_matrix_file,
                destination=tfidf_settings.tfidf_matrix_path,
                missing_ok=True,
            ),
        ]

        load_data_mock.assert_not_called()
        repository_mock.assert_not_called()
        create_lexical_corpus_mock.assert_not_called()
        build_tfidf_index_mock.assert_not_called()
        save_tfidf_index_mock.assert_not_called()

    @pytest.mark.parametrize(
        ("vectorizer_path", "matrix_path"),
        [
            (None, Path("data/artifacts/tfidf_matrix.npz")),
            (
                Path("data/artifacts/tfidf_vectorizer.joblib"),
                None,
            ),
            (None, None),
        ],
    )
    def test_builds_artifacts_when_either_download_is_unavailable(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tfidf_settings: Mock,
        vectorizer_path: Path | None,
        matrix_path: Path | None,
    ) -> None:
        """Build both artifacts when at least one download is unavailable."""
        dataframe = Mock(name="dataframe")
        papers = [
            Mock(name="paper_1"),
            Mock(name="paper_2"),
            Mock(name="paper_3"),
        ]
        texts = [
            "First paper title and abstract",
            "Second paper title and abstract",
            "Third paper title and abstract",
        ]
        vectorizer = Mock(name="vectorizer")
        matrix = csr_matrix((3, 8))

        repository = Mock()
        repository.get_all.return_value = papers

        download_file_mock = Mock(
            side_effect=[
                vectorizer_path,
                matrix_path,
                tfidf_settings.processed_papers_path,
            ]
        )
        load_data_mock = Mock(return_value=dataframe)
        paper_repository_mock = Mock(return_value=repository)
        create_lexical_corpus_mock = Mock(return_value=texts)
        build_tfidf_index_mock = Mock(return_value=(vectorizer, matrix))
        save_tfidf_index_mock = Mock()

        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            download_file_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "load_data",
            load_data_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "PaperRepository",
            paper_repository_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "create_lexical_corpus",
            create_lexical_corpus_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "build_tfidf_index",
            build_tfidf_index_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "save_tfidf_index",
            save_tfidf_index_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(side_effect=[20.0, 25.0]),
        )

        main()

        assert download_file_mock.call_args_list == [
            call(
                repository_id=tfidf_settings.hf_repository,
                remote_filename=(tfidf_settings.hf_tfidf_vectorizer_file),
                destination=tfidf_settings.tfidf_vectorizer_path,
                missing_ok=True,
            ),
            call(
                repository_id=tfidf_settings.hf_repository,
                remote_filename=tfidf_settings.hf_tfidf_matrix_file,
                destination=tfidf_settings.tfidf_matrix_path,
                missing_ok=True,
            ),
            call(
                repository_id=tfidf_settings.hf_repository,
                remote_filename=(tfidf_settings.hf_processed_papers_file),
                destination=tfidf_settings.processed_papers_path,
            ),
        ]

        load_data_mock.assert_called_once_with(
            tfidf_settings.processed_papers_path
        )
        paper_repository_mock.assert_called_once_with(dataframe)
        repository.get_all.assert_called_once_with()
        create_lexical_corpus_mock.assert_called_once_with(papers)
        build_tfidf_index_mock.assert_called_once_with(texts)
        save_tfidf_index_mock.assert_called_once_with(
            vectorizer=vectorizer,
            matrix=matrix,
            vectorizer_path=tfidf_settings.tfidf_vectorizer_path,
            matrix_path=tfidf_settings.tfidf_matrix_path,
        )

    def test_passes_all_repository_papers_to_corpus_builder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tfidf_settings: Mock,
    ) -> None:
        """Pass the exact repository output to lexical-corpus creation."""
        dataframe = Mock(name="dataframe")
        papers = [
            Mock(name="paper_1"),
            Mock(name="paper_2"),
        ]
        texts = [
            "First lexical document",
            "Second lexical document",
        ]
        matrix = csr_matrix((2, 5))

        repository = Mock()
        repository.get_all.return_value = papers
        create_lexical_corpus_mock = Mock(return_value=texts)

        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    None,
                    tfidf_settings.processed_papers_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "load_data",
            Mock(return_value=dataframe),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "PaperRepository",
            Mock(return_value=repository),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "create_lexical_corpus",
            create_lexical_corpus_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "build_tfidf_index",
            Mock(return_value=(Mock(), matrix)),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "save_tfidf_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(side_effect=[1.0, 2.0]),
        )

        main()

        supplied_papers = create_lexical_corpus_mock.call_args.args[0]

        assert supplied_papers is papers

    def test_passes_lexical_corpus_to_index_builder(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tfidf_settings: Mock,
    ) -> None:
        """Pass the exact lexical corpus to TF-IDF index construction."""
        texts = [
            "finite volume solid mechanics",
            "machine learning semantic retrieval",
        ]
        vectorizer = Mock(name="vectorizer")
        matrix = csr_matrix((2, 6))

        repository = Mock()
        repository.get_all.return_value = [Mock(), Mock()]

        build_tfidf_index_mock = Mock(return_value=(vectorizer, matrix))

        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    None,
                    tfidf_settings.processed_papers_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "load_data",
            Mock(return_value=Mock()),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "PaperRepository",
            Mock(return_value=repository),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "create_lexical_corpus",
            Mock(return_value=texts),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "build_tfidf_index",
            build_tfidf_index_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "save_tfidf_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(side_effect=[1.0, 2.0]),
        )

        main()

        supplied_texts = build_tfidf_index_mock.call_args.args[0]

        assert supplied_texts is texts

    def test_logs_existing_artifact_location(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tfidf_settings: Mock,
    ) -> None:
        """Log that the existing TF-IDF artifacts will be used."""
        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            Mock(
                side_effect=[
                    tfidf_settings.tfidf_vectorizer_path,
                    tfidf_settings.tfidf_matrix_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(side_effect=[10.0, 12.0]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_tfidf_index.__name__,
        ):
            main()

        assert "Preparing TF-IDF artifacts." in caplog.messages
        assert (
            "Using the TF-IDF index at "
            f"{tfidf_settings.tfidf_vectorizer_path}." in caplog.messages
        )

    def test_logs_local_build_information(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tfidf_settings: Mock,
    ) -> None:
        """Log corpus size, build progress, and saved matrix shape."""
        papers = [
            Mock(name="paper_1"),
            Mock(name="paper_2"),
            Mock(name="paper_3"),
        ]
        matrix = csr_matrix((3, 12))

        repository = Mock()
        repository.get_all.return_value = papers

        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    None,
                    tfidf_settings.processed_papers_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "load_data",
            Mock(return_value=Mock()),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "PaperRepository",
            Mock(return_value=repository),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "create_lexical_corpus",
            Mock(return_value=["one", "two", "three"]),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "build_tfidf_index",
            Mock(return_value=(Mock(), matrix)),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "save_tfidf_index",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(side_effect=[10.0, 14.0]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_tfidf_index.__name__,
        ):
            main()

        assert (
            "TF-IDF artifacts were not found. Building them locally."
            in caplog.messages
        )
        assert (
            "Create the lexical corpus from 3 processed papers."
            in caplog.messages
        )
        assert (
            "Building the TF-IDF index. "
            "This operation may take a while." in caplog.messages
        )
        assert "Saved TF-IDF index with shape (3, 12)." in caplog.messages

    def test_logs_total_elapsed_time(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        tfidf_settings: Mock,
    ) -> None:
        """Log total preparation time with two decimal places."""
        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            Mock(
                side_effect=[
                    tfidf_settings.tfidf_vectorizer_path,
                    tfidf_settings.tfidf_matrix_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(side_effect=[100.0, 103.456]),
        )

        with caplog.at_level(
            logging.INFO,
            logger=generate_tfidf_index.__name__,
        ):
            main()

        assert (
            "TF-IDF index preparation completed successfully "
            "in 3.46 seconds." in caplog.messages
        )

    def test_propagates_processed_data_download_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tfidf_settings: Mock,
    ) -> None:
        """Propagate errors raised while downloading processed papers."""
        load_data_mock = Mock()

        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    None,
                    RuntimeError("Processed-paper download failed."),
                ]
            ),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "load_data",
            load_data_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="Processed-paper download failed",
        ):
            main()

        load_data_mock.assert_not_called()

    def test_propagates_index_building_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tfidf_settings: Mock,
    ) -> None:
        """Propagate errors raised while building the TF-IDF index."""
        repository = Mock()
        repository.get_all.return_value = [Mock()]
        save_tfidf_index_mock = Mock()

        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    None,
                    tfidf_settings.processed_papers_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "load_data",
            Mock(return_value=Mock()),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "PaperRepository",
            Mock(return_value=repository),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "create_lexical_corpus",
            Mock(return_value=["paper text"]),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "build_tfidf_index",
            Mock(side_effect=RuntimeError("TF-IDF index building failed.")),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "save_tfidf_index",
            save_tfidf_index_mock,
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="TF-IDF index building failed",
        ):
            main()

        save_tfidf_index_mock.assert_not_called()

    def test_propagates_artifact_saving_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tfidf_settings: Mock,
    ) -> None:
        """Propagate errors raised while saving TF-IDF artifacts."""
        repository = Mock()
        repository.get_all.return_value = [Mock()]

        vectorizer = Mock()
        matrix = csr_matrix((1, 4))

        monkeypatch.setattr(
            generate_tfidf_index,
            "get_settings",
            Mock(return_value=tfidf_settings),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "configure_logging",
            Mock(),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "download_file",
            Mock(
                side_effect=[
                    None,
                    None,
                    tfidf_settings.processed_papers_path,
                ]
            ),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "load_data",
            Mock(return_value=Mock()),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "PaperRepository",
            Mock(return_value=repository),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "create_lexical_corpus",
            Mock(return_value=["paper text"]),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "build_tfidf_index",
            Mock(return_value=(vectorizer, matrix)),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "save_tfidf_index",
            Mock(side_effect=RuntimeError("TF-IDF artifact saving failed.")),
        )
        monkeypatch.setattr(
            generate_tfidf_index,
            "perf_counter",
            Mock(return_value=1.0),
        )

        with pytest.raises(
            RuntimeError,
            match="TF-IDF artifact saving failed",
        ):
            main()

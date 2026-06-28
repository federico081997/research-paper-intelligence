"""Tests for downloading files from Hugging Face Hub."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from research_paper_intelligence.storage import huggingface


class TestDownloadFile:
    """Tests for the download_file function."""

    def test_returns_existing_file_without_downloading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Function returns an existing file without downloading."""
        destination = tmp_path / "papers.csv"
        destination.write_text("existing data")

        mock_hf_download = Mock()
        monkeypatch.setattr(
            huggingface,
            "hf_hub_download",
            mock_hf_download,
        )

        result = huggingface.download_file(
            repository_id="test/repository",
            remote_filename="papers.csv",
            destination=destination,
        )

        assert result == destination
        mock_hf_download.assert_not_called()

    def test_downloads_and_copies_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Function downloads and copies a file."""
        cached_path = tmp_path / "cache" / "papers.csv"
        cached_path.parent.mkdir()
        cached_path.write_text("downloaded data")

        destination = tmp_path / "data" / "papers.csv"

        mock_hf_download = Mock(return_value=str(cached_path))
        monkeypatch.setattr(
            huggingface,
            "hf_hub_download",
            mock_hf_download,
        )

        result = huggingface.download_file(
            repository_id="test/repository",
            remote_filename="papers.csv",
            destination=destination,
        )

        assert result == destination
        assert destination.read_text() == "downloaded data"

        mock_hf_download.assert_called_once_with(
            repo_id="test/repository",
            filename="papers.csv",
            repo_type="dataset",
            force_download=False,
        )

    def test_returns_none_when_missing_is_allowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Function returns None when missing is allowed."""

        class FakeRemoteEntryNotFoundError(Exception):
            pass

        monkeypatch.setattr(
            huggingface,
            "RemoteEntryNotFoundError",
            FakeRemoteEntryNotFoundError,
        )
        monkeypatch.setattr(
            huggingface,
            "hf_hub_download",
            Mock(side_effect=FakeRemoteEntryNotFoundError),
        )

        result = huggingface.download_file(
            repository_id="test/repository",
            remote_filename="missing.npy",
            destination=tmp_path / "missing.npy",
            missing_ok=True,
        )

        assert result is None

    def test_raises_when_missing_is_not_allowed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Function raises when missing is not allowed."""

        class FakeRemoteEntryNotFoundError(Exception):
            pass

        monkeypatch.setattr(
            huggingface,
            "RemoteEntryNotFoundError",
            FakeRemoteEntryNotFoundError,
        )
        monkeypatch.setattr(
            huggingface,
            "hf_hub_download",
            Mock(side_effect=FakeRemoteEntryNotFoundError),
        )

        with pytest.raises(FakeRemoteEntryNotFoundError):
            huggingface.download_file(
                repository_id="test/repository",
                remote_filename="missing.csv",
                destination=tmp_path / "missing.csv",
            )

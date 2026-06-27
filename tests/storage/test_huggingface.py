"""Tests for downloading files from Hugging Face Hub."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from research_paper_intelligence.storage import huggingface


class TestDownloadFile:
    """Tests for the ``download_file`` function."""

    @staticmethod
    def test_returns_existing_file_without_downloading(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Skip downloading when the destination already exists."""
        destination = tmp_path / "papers.csv"
        destination.write_text("existing data", encoding="utf-8")

        mock_download = Mock()
        monkeypatch.setattr(huggingface, "hf_hub_download", mock_download)

        result = huggingface.download_file(
            repository_id="user/repository",
            remote_filename="data/papers.csv",
            destination=destination,
        )

        assert result == destination
        mock_download.assert_not_called()

    @staticmethod
    def test_downloads_file_to_destination(
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Download and copy a file to its destination."""
        cached_file = tmp_path / "cached.csv"
        cached_file.write_text("downloaded data", encoding="utf-8")

        destination = tmp_path / "data" / "raw" / "papers.csv"

        mock_download = Mock(return_value=str(cached_file))
        monkeypatch.setattr(huggingface, "hf_hub_download", mock_download)

        result = huggingface.download_file(
            repository_id="user/repository",
            remote_filename="data/papers.csv",
            destination=destination,
        )

        assert result == destination
        assert destination.read_text(encoding="utf-8") == "downloaded data"

        mock_download.assert_called_once_with(
            repo_id="user/repository",
            filename="data/papers.csv",
            repo_type="dataset",
            force_download=False,
        )
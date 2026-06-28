"""Tests performed on the faiss_index_io module."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from research_paper_intelligence.retrieval import faiss_index_io

# -----------------------------------------------------------------------------
#   TestLoadFaissIndex
# -----------------------------------------------------------------------------


class TestLoadFaissIndex:
    """Tests for load_faiss_index function."""

    def test_loads_existing_index(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests that load_faiss_index loads an existing FAISS index."""
        index_path = tmp_path / "papers.faiss"
        index_path.touch()

        mock_index = Mock()
        mock_read_index = Mock(return_value=mock_index)

        monkeypatch.setattr(
            faiss_index_io.faiss,
            "read_index",
            mock_read_index,
        )

        result = faiss_index_io.load_faiss_index(index_path)

        assert result is mock_index
        mock_read_index.assert_called_once_with(str(index_path))

    def test_raises_if_index_does_not_exist(
        self,
        tmp_path: Path,
    ) -> None:
        """Tests that load_faiss_index raises if the index does not exist."""
        index_path = tmp_path / "missing.faiss"

        with pytest.raises(
            FileNotFoundError,
            match=f"FAISS index file not found at {index_path}",
        ):
            faiss_index_io.load_faiss_index(index_path)


# -----------------------------------------------------------------------------
#   TestSaveFaissIndex
# -----------------------------------------------------------------------------


class TestSaveFaissIndex:
    """Tests for save_faiss_index function."""

    def test_saves_index(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tests that save_faiss_index saves the index to the output path."""
        index = Mock()
        output_path = tmp_path / "papers.faiss"
        mock_write_index = Mock()

        monkeypatch.setattr(
            faiss_index_io.faiss,
            "write_index",
            mock_write_index,
        )

        faiss_index_io.save_faiss_index(index, output_path)

        mock_write_index.assert_called_once_with(
            index,
            str(output_path),
        )

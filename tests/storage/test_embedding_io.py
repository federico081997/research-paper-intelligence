"""Tests performed on the test_embedding_io module."""

from pathlib import Path

import numpy as np
import pytest

from research_paper_intelligence.storage.embedding_io import (
    load_embeddings,
    save_embeddings,
)

# -----------------------------------------------------------------------------
#   TestSaveEmbeddings
# -----------------------------------------------------------------------------


class TestSaveEmbeddings:
    """Tests performed on the save_embeddings function."""

    def test_saves_embeddings_to_file(
        self,
        tmp_path: Path,
    ) -> None:
        """Tests that embeddings are saved to a file."""
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        output_path = tmp_path / "embeddings" / "paper_embeddings.npy"

        save_embeddings(embeddings, output_path)

        assert output_path.exists()

        saved_embeddings = np.load(output_path)
        np.testing.assert_array_equal(embeddings, saved_embeddings)


# -----------------------------------------------------------------------------
#   TestLoadEmbeddings
# -----------------------------------------------------------------------------


class TestLoadEmbeddings:
    """Tests performed on the load_embeddings function."""

    def test_load_embeddings_returns_float32_array(
        self,
        tmp_path: Path,
    ) -> None:
        """Tests that the embeddings are loaded as float32."""
        embeddings = np.array(
            [[1.0, 2.0], [3.0, 4.0]],
            dtype=np.float64,
        )
        embeddings_path = tmp_path / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        result = load_embeddings(embeddings_path)

        expected = embeddings.astype(np.float32)

        np.testing.assert_array_equal(result, expected)
        assert result.dtype == np.float32

    def test_load_embeddings_raises_when_file_is_missing(
        self,
        tmp_path: Path,
    ) -> None:
        """Tests that load_embeddings raises when the file is missing."""
        missing_path = tmp_path / "missing_embeddings.npy"

        with pytest.raises(
            FileNotFoundError,
            match=f"Embeddings were not found at: {missing_path}",
        ):
            load_embeddings(missing_path)

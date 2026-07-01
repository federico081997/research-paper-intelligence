"""Tests performed on the faiss_indexing module."""

import faiss
import numpy as np

from research_paper_intelligence.retrieval.faiss_index_builder import (
    build_faiss_index,
)

# -----------------------------------------------------------------------------
#   TestBuildFaissIndex
# -----------------------------------------------------------------------------


class TestBuildFaissIndex:
    """Tests for build_faiss_index function."""

    def test_builds_populated_index(self) -> None:
        """Tests that build_faiss_index function builds a populated index."""
        embeddings = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )

        index = build_faiss_index(embeddings)

        assert isinstance(index, faiss.IndexFlatIP)
        assert index.d == embeddings.shape[1]
        assert index.ntotal == embeddings.shape[0]

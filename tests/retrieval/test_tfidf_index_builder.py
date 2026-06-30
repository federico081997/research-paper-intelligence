"""Tests for TF-IDF index construction."""

from datetime import date
from unittest.mock import Mock, call

import pytest
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

import research_paper_intelligence.retrieval.tfidf_index_builder as script
from research_paper_intelligence.domain.paper import Paper


@pytest.fixture
def paper() -> Paper:
    """Return one paper for testing."""
    return Paper(
        paper_id="paper-001",
        title="Finite Volume Methods",
        abstract="A paper about numerical methods.",
        authors="Alice Smith, Bob Jones",
        category="Computational Engineering",
        published_date=date(2024, 1, 15),
    )


@pytest.fixture
def papers() -> list[Paper]:
    """Return multiple papers for testing."""
    return [
        Paper(
            paper_id="paper-001",
            title="Finite Volume Methods",
            abstract="A paper about numerical methods.",
            authors="Alice Smith, Bob Jones",
            category="Computational Engineering",
            published_date=date(2024, 1, 15),
        ),
        Paper(
            paper_id="paper-002",
            title="Machine Learning for Engineering",
            abstract="A paper about machine learning.",
            authors="Carol Brown",
            category="Machine Learning",
            published_date=date(2025, 6, 20),
        ),
    ]


class TestCreateLexicalText:
    """Tests for create_lexical_text function."""

    def test_creates_text_with_title_repeated(
        self,
        paper: Paper,
    ) -> None:
        """Function to create lexical text from a paper."""
        result = script.create_lexical_text(paper)

        assert result == (
            "Finite Volume Methods Finite Volume Methods "
            "A paper about numerical methods. Alice Smith, Bob Jones"
        )


class TestCreateLexicalCorpus:
    """Tests for create_lexical_corpus function."""

    def test_creates_text_for_each_paper(
        self,
        monkeypatch: pytest.MonkeyPatch,
        papers: list[Paper],
    ) -> None:
        """Function to create lexical corpus from a list of papers."""
        mock_lexical_text = Mock(
            side_effect=[
                "A paper about numerical methods.",
                "A paper about machine learning.",
            ]
        )
        monkeypatch.setattr(
            script,
            "create_lexical_text",
            mock_lexical_text,
        )

        result = script.create_lexical_corpus(papers)

        assert result == [
            "A paper about numerical methods.",
            "A paper about machine learning.",
        ]
        assert mock_lexical_text.call_args_list == [
            call(papers[0]),
            call(papers[1]),
        ]


class TestBuildTfidfIndex:
    """Tests for build_tfidf_index function."""

    def test_builds_vectorizer_and_matrix(self) -> None:
        """Tests that it creates a TF-IDF vectorizer and matrix."""
        texts = [
            "finite volume computational mechanics",
            "machine learning semantic retrieval",
            "plasticity numerical solver development",
        ]

        vectorizer, matrix = script.build_tfidf_index(texts)

        assert isinstance(vectorizer, TfidfVectorizer)
        assert isinstance(matrix, csr_matrix)
        assert matrix.shape[0] == len(texts)
        assert matrix.shape[1] > 0

    def test_raises_error_for_empty_texts(self) -> None:
        """Tests that it raises an error for empty texts."""
        with pytest.raises(
            ValueError,
            match="At least one paper text is required.",
        ):
            script.build_tfidf_index([])

    def test_raises_error_for_blank_text(self) -> None:
        """Tests that it raises an error for blank text."""
        with pytest.raises(
            ValueError,
            match="At least one paper text is empty or whitespace only.",
        ):
            script.build_tfidf_index(["Valid paper text", "   "])

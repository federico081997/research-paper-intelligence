"""Tests for the research-assistant paper retriever."""

from datetime import date
from unittest.mock import Mock

import pytest

from research_paper_intelligence.assistant.models import RetrievedPaper
from research_paper_intelligence.assistant.retrieval import AssistantRetriever
from research_paper_intelligence.services.search_service import SearchService


@pytest.fixture
def assistant_search_service() -> Mock:
    """Create a mocked search service for assistant retrieval tests."""
    return Mock(spec=SearchService)


@pytest.fixture
def assistant_retriever(
    assistant_search_service: Mock,
) -> AssistantRetriever:
    """Create an assistant retriever using a mocked search service."""
    return AssistantRetriever(search_service=assistant_search_service)


@pytest.fixture
def assistant_search_results() -> list[Mock]:
    """Create representative search-service results."""
    first_paper = Mock()
    first_paper.paper_id = "2401.12345"
    first_paper.title = "Finite volume methods for solid mechanics"
    first_paper.abstract = (
        "A block-coupled finite volume method is developed "
        "for computational solid mechanics."
    )
    first_paper.authors = "Author One, Author Two"
    first_paper.category = "Computational Engineering"
    first_paper.published_date = date(2025, 1, 15)

    first_result = Mock()
    first_result.paper = first_paper
    first_result.rank = 1
    first_result.hybrid_score = 0.91

    second_paper = Mock()
    second_paper.paper_id = "2402.67890"
    second_paper.title = "Elastoplasticity using finite volume discretisation"
    second_paper.abstract = (
        "This paper studies elastoplastic constitutive models "
        "within a finite volume framework."
    )
    second_paper.authors = "Author Three"
    second_paper.category = "Computational Mechanics"
    second_paper.published_date = date(2024, 6, 10)

    second_result = Mock()
    second_result.paper = second_paper
    second_result.rank = 2
    second_result.hybrid_score = 0.84

    return [first_result, second_result]


class TestAssistantRetriever:
    """Tests for the AssistantRetriever class."""

    def test_passes_query_and_result_count_to_search_service(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
    ) -> None:
        """Pass the query and requested result count to search."""
        assistant_search_service.search.return_value = []

        assistant_retriever.retrieve(
            query="finite volume elastoplasticity",
            result_k=7,
        )

        assistant_search_service.search.assert_called_once_with(
            "finite volume elastoplasticity",
            7,
        )

    def test_converts_search_results_to_retrieved_papers(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
        assistant_search_results: list[Mock],
    ) -> None:
        """Convert search-service results into retrieved-paper models."""
        assistant_search_service.search.return_value = assistant_search_results

        result = assistant_retriever.retrieve(
            query="finite volume solid mechanics",
            result_k=2,
        )

        assert result == [
            RetrievedPaper(
                paper_id="2401.12345",
                title="Finite volume methods for solid mechanics",
                abstract=(
                    "A block-coupled finite volume method is developed "
                    "for computational solid mechanics."
                ),
                authors="Author One, Author Two",
                category="Computational Engineering",
                published_date=date(2025, 1, 15),
                rank=1,
                hybrid_score=0.91,
            ),
            RetrievedPaper(
                paper_id="2402.67890",
                title=("Elastoplasticity using finite volume discretisation"),
                abstract=(
                    "This paper studies elastoplastic constitutive models "
                    "within a finite volume framework."
                ),
                authors="Author Three",
                category="Computational Mechanics",
                published_date=date(2024, 6, 10),
                rank=2,
                hybrid_score=0.84,
            ),
        ]

    def test_returns_retrieved_paper_instances(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
        assistant_search_results: list[Mock],
    ) -> None:
        """Return RetrievedPaper instances rather than search results."""
        assistant_search_service.search.return_value = assistant_search_results

        result = assistant_retriever.retrieve(
            query="finite volume methods",
            result_k=2,
        )

        assert all(isinstance(paper, RetrievedPaper) for paper in result)

    def test_preserves_search_result_order(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
        assistant_search_results: list[Mock],
    ) -> None:
        """Preserve the ordering produced by the search service."""
        assistant_search_service.search.return_value = assistant_search_results

        result = assistant_retriever.retrieve(
            query="finite volume methods",
            result_k=2,
        )

        assert [paper.paper_id for paper in result] == [
            "2401.12345",
            "2402.67890",
        ]
        assert [paper.rank for paper in result] == [1, 2]

    def test_maps_all_required_paper_fields(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
        assistant_search_results: list[Mock],
    ) -> None:
        """Map all assistant context fields from a search result."""
        assistant_search_service.search.return_value = [
            assistant_search_results[0]
        ]

        result = assistant_retriever.retrieve(
            query="finite volume methods",
            result_k=1,
        )

        paper = result[0]

        assert paper.paper_id == "2401.12345"
        assert paper.title == ("Finite volume methods for solid mechanics")
        assert paper.abstract == (
            "A block-coupled finite volume method is developed "
            "for computational solid mechanics."
        )
        assert paper.authors == "Author One, Author Two"
        assert paper.category == "Computational Engineering"
        assert paper.published_date == date(2025, 1, 15)
        assert paper.rank == 1
        assert paper.hybrid_score == pytest.approx(0.91)

    def test_returns_empty_list_when_search_finds_no_results(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
    ) -> None:
        """Return an empty list when the search service finds no papers."""
        assistant_search_service.search.return_value = []

        result = assistant_retriever.retrieve(
            query="unavailable scientific topic",
            result_k=5,
        )

        assert result == []

    def test_invokes_search_service_once(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
    ) -> None:
        """Invoke the underlying search service exactly once."""
        assistant_search_service.search.return_value = []

        assistant_retriever.retrieve(
            query="semantic search",
            result_k=5,
        )

        assistant_search_service.search.assert_called_once()

    def test_creates_new_retrieved_paper_objects(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
        assistant_search_results: list[Mock],
    ) -> None:
        """Create new assistant-specific objects from domain results."""
        assistant_search_service.search.return_value = assistant_search_results

        result = assistant_retriever.retrieve(
            query="finite volume methods",
            result_k=2,
        )

        assert result[0] is not assistant_search_results[0]
        assert result[1] is not assistant_search_results[1]
        assert result[0] is not assistant_search_results[0].paper
        assert result[1] is not assistant_search_results[1].paper

    def test_propagates_search_service_errors(
        self,
        assistant_retriever: AssistantRetriever,
        assistant_search_service: Mock,
    ) -> None:
        """Propagate errors raised by the underlying search service."""
        assistant_search_service.search.side_effect = RuntimeError(
            "Paper search failed."
        )

        with pytest.raises(
            RuntimeError,
            match="Paper search failed",
        ):
            assistant_retriever.retrieve(
                query="finite volume methods",
                result_k=5,
            )

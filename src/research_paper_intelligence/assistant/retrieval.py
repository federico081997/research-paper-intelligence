"""Retrieval functions used by the research assistant."""

from research_paper_intelligence.assistant.models import RetrievedPaper
from research_paper_intelligence.services.search_service import SearchService


class AssistantRetriever:
    """Retrieve paper context using the existing search service."""

    def __init__(
        self,
        search_service: SearchService,
    ):
        """Initialize the retriever.

        Args:
            search_service: the search service used to retrieve papers.
        """
        self._search_service = search_service

    def retrieve(
        self,
        query: str,
        result_k: int,
    ) -> list[RetrievedPaper]:
        """Retrieve papers relevant to a question.

        Args:
            query: sentence used to retrieve papers.
            result_k: the number of results to return.
        """
        results = self._search_service.search(query, result_k)

        return [
            RetrievedPaper(
                paper_id=result.paper.paper_id,
                title=result.paper.title,
                abstract=result.paper.abstract,
                authors=result.paper.authors,
                category=result.paper.category,
                published_date=result.paper.published_date,
                rank=result.rank,
                hybrid_score=result.hybrid_score,
            )
            for result in results
        ]

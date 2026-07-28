"""Tests for the research-assistant workflow models."""

from dataclasses import FrozenInstanceError
from datetime import date

import pytest
from pydantic import ValidationError

from research_paper_intelligence.assistant.models import (
    DirectAnswer,
    PlanRequest,
    ResearchAnswer,
    RetrievalGrader,
    RetrievedPaper,
    RewrittenQuery,
)


class TestRetrievedPaper:
    """Tests for the RetrievedPaper dataclass."""

    def test_creates_retrieved_paper(self) -> None:
        """Create a retrieved paper containing the supplied metadata."""
        paper = RetrievedPaper(
            paper_id="2401.12345",
            title="Finite volume methods",
            abstract="An abstract about finite volume methods.",
            authors="Author One, Author Two",
            category="Computational Engineering",
            published_date=date(2025, 1, 15),
            rank=1,
            hybrid_score=0.84,
        )

        assert paper.paper_id == "2401.12345"
        assert paper.title == "Finite volume methods"
        assert paper.abstract == ("An abstract about finite volume methods.")
        assert paper.authors == "Author One, Author Two"
        assert paper.category == "Computational Engineering"
        assert paper.published_date == date(2025, 1, 15)
        assert paper.rank == 1
        assert paper.hybrid_score == pytest.approx(0.84)

    def test_compares_papers_by_field_values(self) -> None:
        """Treat papers with identical field values as equal."""
        first_paper = RetrievedPaper(
            paper_id="2401.12345",
            title="Finite volume methods",
            abstract="Paper abstract.",
            authors="Author One",
            category="Engineering",
            published_date=date(2025, 1, 15),
            rank=1,
            hybrid_score=0.84,
        )
        second_paper = RetrievedPaper(
            paper_id="2401.12345",
            title="Finite volume methods",
            abstract="Paper abstract.",
            authors="Author One",
            category="Engineering",
            published_date=date(2025, 1, 15),
            rank=1,
            hybrid_score=0.84,
        )

        assert first_paper == second_paper

    def test_is_immutable(self) -> None:
        """Prevent retrieved-paper fields from being modified."""
        paper = RetrievedPaper(
            paper_id="2401.12345",
            title="Original title",
            abstract="Paper abstract.",
            authors="Author One",
            category="Engineering",
            published_date=date(2025, 1, 15),
            rank=1,
            hybrid_score=0.84,
        )

        with pytest.raises(FrozenInstanceError):
            paper.title = "Updated title"  # type: ignore[misc]


class TestPlanRequest:
    """Tests for the PlanRequest model."""

    def test_accepts_valid_direct_request(self) -> None:
        """Accept a direct request without retrieval parameters."""
        plan = PlanRequest(
            request_type="direct",
            search_query="",
            result_k=0,
        )

        assert plan.request_type == "direct"
        assert plan.search_query == ""
        assert plan.result_k == 0

    @pytest.mark.parametrize("result_k", [1, 5, 10])
    def test_accepts_valid_retrieval_request(
        self,
        result_k: int,
    ) -> None:
        """Accept valid retrieval requests within the result limit."""
        plan = PlanRequest(
            request_type="retrieval",
            search_query="finite volume elastoplasticity",
            result_k=result_k,
        )

        assert plan.request_type == "retrieval"
        assert plan.search_query == "finite volume elastoplasticity"
        assert plan.result_k == result_k

    def test_rejects_direct_request_with_search_query(self) -> None:
        """Reject a direct request containing a search query."""
        with pytest.raises(
            ValidationError,
            match="Direct requests must have an empty search query",
        ):
            PlanRequest(
                request_type="direct",
                search_query="machine learning",
                result_k=0,
            )

    @pytest.mark.parametrize("result_k", [1, 5, 10])
    def test_rejects_direct_request_with_nonzero_result_k(
        self,
        result_k: int,
    ) -> None:
        """Reject a direct request containing a nonzero result count."""
        with pytest.raises(
            ValidationError,
            match="Direct requests must have result_k=0",
        ):
            PlanRequest(
                request_type="direct",
                search_query="",
                result_k=result_k,
            )

    def test_rejects_retrieval_request_without_search_query(
        self,
    ) -> None:
        """Reject a retrieval request without a search query."""
        with pytest.raises(
            ValidationError,
            match="Retrieval requests must have a search query",
        ):
            PlanRequest(
                request_type="retrieval",
                search_query="",
                result_k=5,
            )

    def test_rejects_retrieval_request_with_zero_results(
        self,
    ) -> None:
        """Reject a retrieval request requesting zero papers."""
        with pytest.raises(
            ValidationError,
            match=("Retrieval requests must have result_k of at least 1"),
        ):
            PlanRequest(
                request_type="retrieval",
                search_query="machine learning",
                result_k=0,
            )

    def test_rejects_negative_result_k(self) -> None:
        """Reject result counts below the field minimum."""
        with pytest.raises(
            ValidationError,
            match="greater than or equal to 0",
        ):
            PlanRequest(
                request_type="direct",
                search_query="",
                result_k=-1,
            )

    def test_rejects_result_k_above_maximum(self) -> None:
        """Reject result counts greater than ten."""
        with pytest.raises(
            ValidationError,
            match="less than or equal to 10",
        ):
            PlanRequest(
                request_type="retrieval",
                search_query="machine learning",
                result_k=11,
            )

    def test_rejects_unknown_request_type(self) -> None:
        """Reject request types outside the permitted literals."""
        with pytest.raises(ValidationError):
            PlanRequest.model_validate(
                {
                    "request_type": "unknown",
                    "search_query": "",
                    "result_k": 0,
                }
            )


class TestDirectAnswer:
    """Tests for the DirectAnswer model."""

    def test_accepts_nonempty_answer(self) -> None:
        """Accept a nonempty direct answer."""
        answer = DirectAnswer(final_answer="Hello. How can I help you?")

        assert answer.final_answer == "Hello. How can I help you?"

    def test_rejects_empty_answer(self) -> None:
        """Reject an empty direct answer."""
        with pytest.raises(
            ValidationError,
            match="at least 1 character",
        ):
            DirectAnswer(final_answer="")


class TestRetrievalGrader:
    """Tests for the RetrievalGrader model."""

    @pytest.mark.parametrize(
        ("retrieval_sufficient", "retrieval_feedback"),
        [
            (
                True,
                "The papers contain enough relevant evidence.",
            ),
            (
                False,
                "The papers do not address the requested method.",
            ),
        ],
    )
    def test_accepts_retrieval_assessment(
        self,
        retrieval_sufficient: bool,
        retrieval_feedback: str,
    ) -> None:
        """Store the retrieval decision and supporting feedback."""
        grader = RetrievalGrader(
            retrieval_sufficient=retrieval_sufficient,
            retrieval_feedback=retrieval_feedback,
        )

        assert grader.retrieval_sufficient is retrieval_sufficient
        assert grader.retrieval_feedback == retrieval_feedback

    def test_rejects_missing_retrieval_decision(self) -> None:
        """Reject output without a retrieval-sufficiency decision."""
        with pytest.raises(ValidationError):
            RetrievalGrader.model_validate(
                {
                    "retrieval_feedback": (
                        "The retrieved evidence is sufficient."
                    )
                }
            )

    def test_rejects_missing_retrieval_feedback(self) -> None:
        """Reject output without retrieval feedback."""
        with pytest.raises(ValidationError):
            RetrievalGrader.model_validate({"retrieval_sufficient": True})


class TestRewrittenQuery:
    """Tests for the RewrittenQuery model."""

    def test_accepts_nonempty_search_query(self) -> None:
        """Accept a nonempty rewritten search query."""
        rewritten_query = RewrittenQuery(
            search_query=("block-coupled finite volume elastoplasticity")
        )

        assert rewritten_query.search_query == (
            "block-coupled finite volume elastoplasticity"
        )

    def test_rejects_empty_search_query(self) -> None:
        """Reject an empty rewritten search query."""
        with pytest.raises(
            ValidationError,
            match="at least 1 character",
        ):
            RewrittenQuery(search_query="")


class TestResearchAnswer:
    """Tests for the ResearchAnswer model."""

    def test_accepts_nonempty_research_answer(self) -> None:
        """Accept a nonempty Markdown research answer."""
        final_answer = (
            "## Findings\n\n"
            "- Finite volume methods conserve fluxes.\n\n"
            "## References\n\n"
            "[1] Example paper."
        )

        answer = ResearchAnswer(final_answer=final_answer)

        assert answer.final_answer == final_answer

    def test_rejects_empty_research_answer(self) -> None:
        """Reject an empty research answer."""
        with pytest.raises(
            ValidationError,
            match="at least 1 character",
        ):
            ResearchAnswer(final_answer="")

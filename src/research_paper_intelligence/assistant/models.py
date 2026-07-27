"""Models used by the research assistant workflow."""

from dataclasses import dataclass
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


@dataclass(frozen=True)
class RetrievedPaper:
    """Paper context supplied to the large language model."""

    paper_id: str
    title: str
    abstract: str
    authors: str
    category: str
    published_date: date
    rank: int
    hybrid_score: float


class PlanRequest(BaseModel):
    """Structured output produced by the request-planning model."""

    request_type: Literal["direct", "retrieval"] = Field(
        description=(
            "Whether the latest user request requires retrieving "
            "research papers."
        ),
    )

    search_query: str = Field(
        description=(
            "A standalone scientific-paper search query when request_type "
            "is 'retrieval', otherwise an empty string."
        ),
    )

    result_k: int = Field(
        ge=0,
        le=10,
        description=(
            "Number of papers to retrieve. Zero for direct requests."
        ),
    )

    @model_validator(mode="after")
    def validate_plan(self) -> Self:
        """Validate consistency between request type and retrieval fields."""
        if self.request_type == "direct":
            if self.search_query:
                raise ValueError(
                    "Direct requests must have an empty search query."
                )

            if self.result_k != 0:
                raise ValueError("Direct requests must have result_k=0.")

        if self.request_type == "retrieval":
            if not self.search_query:
                raise ValueError(
                    "Retrieval requests must have a search query."
                )

            if self.result_k < 1:
                raise ValueError(
                    "Retrieval requests must have result_k of at least 1."
                )

        return self


class DirectAnswer(BaseModel):
    """Structured output produced by the direct-answer model."""

    final_answer: str = Field(
        min_length=1,
        description=(
            "A clear natural-language response to the user's request. "
            "The response must not claim that research papers were retrieved."
        ),
    )


class RetrievalGrader(BaseModel):
    """Structured output produced by the grade retrieval model."""

    retrieval_sufficient: bool = Field(
        description=(
            "Whether the retrieved papers contain enough relevant evidence "
            "to answer the user's latest request."
        ),
    )

    retrieval_feedback: str = Field(
        description=(
            "A concise explanation of why the retrieved evidence is "
            "sufficient, or what information is missing when it is "
            "insufficient."
        ),
    )


class RewrittenQuery(BaseModel):
    """Structured output produced by the query-rewriting model."""

    search_query: str = Field(
        min_length=1,
        description=(
            "A revised standalone query suitable for searching a "
            "scientific-paper collection."
        ),
    )


class ResearchAnswer(BaseModel):
    """Structured output produced by a research-answer model."""

    final_answer: str = Field(
        min_length=1,
        description=(
            "The final response to the user. "
            "Use valid Markdown formatting. "
            "Separate paragraphs, headings, lists, and the references section "
            "with actual blank lines. "
            "Do not include literal escaped newline sequences such as "
            "`\\n` or `\\n\\n` in the response. "
            "Use Markdown headings with `##` or `###`, and use `-` for "
            "unordered list items. "
            "If the user explicitly requested more than 10 papers, explicitly "
            "state that the system can retrieve a maximum of 10 papers at "
            "a time."
        ),
    )

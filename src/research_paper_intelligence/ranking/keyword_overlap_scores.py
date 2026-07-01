"""Utilities for calculating keyword overlap scores."""

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer

from research_paper_intelligence.domain.paper import Paper


def calculate_keyword_overlap_scores(
    query: str,
    candidate_papers: Sequence[Paper],
    vectorizer: TfidfVectorizer,
) -> NDArray[np.float64]:
    """Calculate keyword-overlap scores for candidate papers.

    Args:
        query: The query string.
        candidate_papers: The sequence of candidate papers.
        vectorizer: The vectorizer that is used to extract keywords.

    Returns:
        The keyword-overlap scores for the query and candidate papers.
    """
    analyzer = vectorizer.build_analyzer()
    query_keywords = set(analyzer(query))

    if not query_keywords:
        return np.zeros(len(candidate_papers))

    scores = np.empty(len(candidate_papers))
    for i, paper in enumerate(candidate_papers):
        paper_text = paper.title + " " + paper.abstract
        paper_keywords = set(analyzer(paper_text))

        overlap = len(query_keywords.intersection(paper_keywords)) / len(
            query_keywords
        )
        scores[i] = overlap

    return scores

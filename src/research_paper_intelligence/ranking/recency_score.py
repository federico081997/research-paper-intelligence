"""Utility functions for calculating the recency score."""

from collections.abc import Sequence
from datetime import date
from math import exp, log

import numpy as np
from numpy.typing import NDArray

from research_paper_intelligence.domain.paper import Paper


def calculate_recency_score(
    published_date: date, half_life_years: float
) -> float:
    """Return a recency score between 0 and 1.

    Args:
        published_date: The date the paper was published.
        half_life_years: The half-life of the recency score in years.
    """
    if half_life_years <= 0:
        raise ValueError("Half-life years must be positive.")

    age_years = max(0, (date.today() - published_date).days) / 365.25
    decay_rate = log(2) / half_life_years

    return exp(-decay_rate * age_years)


def calculate_recency_scores(
    candidate_papers: Sequence[Paper],
    half_life_years: float,
) -> NDArray[np.float64]:
    """Calculate the recency scores for a number of candidate papers.

    Args:
        candidate_papers: The list of candidate papers.
        half_life_years: The half-life of the recency score in years.
    """
    return np.array(
        [
            calculate_recency_score(
                candidate_paper.published_date, half_life_years
            )
            for candidate_paper in candidate_papers
        ]
    )

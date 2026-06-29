"""Defines the domain model for a research paper, containing metadata."""

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Paper:
    """Represents a research paper."""

    paper_id: str
    title: str
    abstract: str
    authors: str
    category: str
    published_date: datetime

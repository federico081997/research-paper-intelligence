"""Build the TF-IDF representation of the research papers."""

from collections.abc import Iterable

from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from research_paper_intelligence.domain.paper import Paper


def create_lexical_text(paper: Paper) -> str:
    """Create the text representation of a paper for TF-IDF.

    Args:
        paper: The paper to create the text representation for.
    """
    return " ".join(
        [
            paper.title,
            paper.title,  # Gives title terms more influence.
            paper.abstract,
            paper.authors,
        ]
    )


def create_lexical_corpus(
    papers: Iterable[Paper],
) -> list[str]:
    """Create lexical texts for a collection of papers.

    Args:
        papers: The papers to create the lexical texts for.
    """
    return [create_lexical_text(paper) for paper in papers]


def build_tfidf_index(
    texts: Iterable[str],
) -> tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer and transform the paper texts.

    Args:
        texts: The texts on which to fit the TF-IDF vectorizer.

    Returns:
        The fitted TF-IDF vectorizer and the transformed texts.

    Raises:
        ValueError: If the texts are empty.
    """
    texts_list = list(texts)

    if not texts_list:
        raise ValueError("At least one paper text is required.")

    if any(not text.strip() for text in texts_list):
        raise ValueError(
            "At least one paper text is empty or whitespace only."
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
        norm="l2",
    )

    matrix = vectorizer.fit_transform(texts_list)

    return vectorizer, matrix

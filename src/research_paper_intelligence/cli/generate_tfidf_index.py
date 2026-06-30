"""Download or build the TF-IDF retrieval artifacts."""

import logging
from time import perf_counter

from research_paper_intelligence.config import get_settings
from research_paper_intelligence.data.data_loader import load_data
from research_paper_intelligence.logging_config import configure_logging
from research_paper_intelligence.repositories.paper_repository import (
    PaperRepository,
)
from research_paper_intelligence.retrieval.tfidf_index_builder import (
    build_tfidf_index,
    create_lexical_corpus,
)
from research_paper_intelligence.storage.huggingface import download_file
from research_paper_intelligence.storage.tfidf_index_io import (
    save_tfidf_index,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Download existing TF-IDF artifacts or build them locally."""
    start_time = perf_counter()

    settings = get_settings()
    configure_logging(settings)

    logger.info("Preparing TF-IDF artifacts.")

    # Try to download both artifacts from Hugging Face.
    vectorizer_path = download_file(
        repository_id=settings.hf_repository,
        remote_filename=settings.hf_tfidf_vectorizer_file,
        destination=settings.tfidf_vectorizer_path,
        missing_ok=True,
    )

    matrix_path = download_file(
        repository_id=settings.hf_repository,
        remote_filename=settings.hf_tfidf_matrix_file,
        destination=settings.tfidf_matrix_path,
        missing_ok=True,
    )

    if vectorizer_path is None or matrix_path is None:
        logger.warning(
            "TF-IDF artifacts were not found. Building them locally."
        )

        download_file(
            repository_id=settings.hf_repository,
            remote_filename=settings.hf_processed_papers_file,
            destination=settings.processed_papers_path,
        )

        dataframe = load_data(settings.processed_papers_path)

        # Extract papers from the repository
        repository = PaperRepository(dataframe)
        papers = repository.get_all()

        # Create the lexical text representation.
        logger.info(
            "Create the lexical corpus from %d processed papers.", len(papers)
        )
        texts = create_lexical_corpus(papers)

        logger.info(
            "Building the TF-IDF index. This operation may take a while."
        )

        # Fit the vectorizer and create the sparse paper matrix.
        vectorizer, matrix = build_tfidf_index(texts)

        # Save the TF-IDF artifacts.
        save_tfidf_index(
            vectorizer=vectorizer,
            matrix=matrix,
            vectorizer_path=settings.tfidf_vectorizer_path,
            matrix_path=settings.tfidf_matrix_path,
        )

        logger.info(
            "Saved TF-IDF index with shape %s.",
            matrix.shape,
        )
    else:
        logger.info(
            "Using the TF-IDF index at %s.",
            settings.tfidf_vectorizer_path,
        )

    logger.info(
        "TF-IDF index preparation completed successfully in %.2f seconds.",
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()

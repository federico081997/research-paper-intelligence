"""Generate embeddings for the processed research papers."""

import logging
from time import perf_counter

from research_paper_intelligence.config import get_settings
from research_paper_intelligence.embeddings.embedding_pipeline import (
    run_embedding_pipeline,
)
from research_paper_intelligence.logging_config import configure_logging
from research_paper_intelligence.storage.huggingface import download_file

logger = logging.getLogger(__name__)


def main() -> None:
    """Download existing embeddings or generate them locally."""
    start_time = perf_counter()

    settings = get_settings()
    configure_logging(settings)

    logger.info("Preparing paper embeddings.")

    embeddings_path = download_file(
        repository_id=settings.hf_repository,
        remote_filename=settings.hf_paper_embeddings_file,
        destination=settings.paper_embeddings_path,
        missing_ok=True,
    )

    if embeddings_path is None:
        logger.info(
            "Precomputed paper embeddings are unavailable. "
            "Generating them locally from the processed papers."
        )

        download_file(
            repository_id=settings.hf_repository,
            remote_filename=settings.hf_processed_papers_file,
            destination=settings.processed_papers_path,
        )

        logger.info(
            "Generating paper embeddings from %s.",
            settings.processed_papers_path,
        )

        run_embedding_pipeline(settings)

        logger.info(
            "Paper embeddings were generated and saved to %s.",
            settings.paper_embeddings_path,
        )
    else:
        logger.info(
            "Using the existing paper embeddings at %s.",
            embeddings_path,
        )

    logger.info(
        "Paper embedding preparation completed successfully in %.2f seconds.",
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()

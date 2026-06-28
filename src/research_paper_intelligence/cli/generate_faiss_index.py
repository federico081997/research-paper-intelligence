"""Download an existing FAISS index or build it locally."""

import logging
from time import perf_counter

from research_paper_intelligence.config import get_settings
from research_paper_intelligence.embeddings.embedding_io import (
    load_embeddings,
)
from research_paper_intelligence.logging_config import configure_logging
from research_paper_intelligence.retrieval.faiss_index_builder import (
    build_faiss_index,
)
from research_paper_intelligence.retrieval.faiss_index_io import (
    save_faiss_index,
)
from research_paper_intelligence.storage.huggingface import download_file

logger = logging.getLogger(__name__)


def main() -> None:
    """Download an existing FAISS index or build it from embeddings."""
    start_time = perf_counter()

    settings = get_settings()
    configure_logging(settings)

    logger.info("Preparing the FAISS index.")

    faiss_index_path = download_file(
        repository_id=settings.hf_repository,
        remote_filename=settings.hf_faiss_index_papers_file,
        destination=settings.faiss_index_papers_path,
        missing_ok=True,
    )

    if faiss_index_path is None:
        logger.info(
            "A precomputed FAISS index is unavailable. "
            "Building the index locally from paper embeddings."
        )

        download_file(
            repository_id=settings.hf_repository,
            remote_filename=settings.hf_paper_embeddings_file,
            destination=settings.paper_embeddings_path,
        )

        embeddings = load_embeddings(settings.paper_embeddings_path)

        logger.info(
            "Building a FAISS index from %d embeddings.",
            embeddings.shape[0],
        )

        index = build_faiss_index(embeddings)

        save_faiss_index(
            index=index,
            output_path=settings.faiss_index_papers_path,
        )

        logger.info(
            "Saved the FAISS index containing %d vectors to %s.",
            index.ntotal,
            settings.faiss_index_papers_path,
        )
    else:
        logger.info(
            "Using the existing FAISS index at %s.",
            faiss_index_path,
        )

    logger.info(
        "FAISS index preparation completed successfully in %.2f seconds.",
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()

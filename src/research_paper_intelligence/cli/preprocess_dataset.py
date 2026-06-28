"""Main script for running the research-paper processing pipeline."""

import logging
from time import perf_counter

from research_paper_intelligence.config import get_settings
from research_paper_intelligence.data.preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
from research_paper_intelligence.logging_config import configure_logging
from research_paper_intelligence.storage.huggingface import download_file

logger = logging.getLogger(__name__)


def main() -> None:
    """Download raw papers and run the preprocessing pipeline."""
    start_time = perf_counter()

    settings = get_settings()
    configure_logging(settings)

    logger.info("Preparing processed research-paper data.")

    download_file(
        repository_id=settings.hf_repository,
        remote_filename=settings.hf_raw_papers_file,
        destination=settings.raw_papers_path,
    )

    logger.info(
        "Preprocessing raw papers from %s.",
        settings.raw_papers_path,
    )

    run_preprocessing_pipeline(settings)

    logger.info(
        "Processed papers were saved to %s.",
        settings.processed_papers_path,
    )

    logger.info(
        "Research-paper preprocessing completed successfully in %.2f seconds.",
        perf_counter() - start_time,
    )


if __name__ == "__main__":
    main()

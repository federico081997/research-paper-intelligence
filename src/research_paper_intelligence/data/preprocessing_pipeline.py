"""Coordinate the research-paper dataset preprocessing workflow."""

import logging
from time import perf_counter

from research_paper_intelligence.config import Settings
from research_paper_intelligence.data.data_loader import load_data, save_to_csv
from research_paper_intelligence.data.preprocessing import preprocess_dataset

logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(settings: Settings) -> None:
    """Run the research-paper preprocessing pipeline."""
    start_time = perf_counter()

    logger.info("Starting research-paper preprocessing pipeline")
    logger.debug("Raw dataset path: %s", settings.raw_papers_path)
    logger.debug(
        "Processed dataset path: %s",
        settings.processed_papers_path,
    )

    # Load the raw dataset.
    logger.info("Loading raw dataset from %s", settings.raw_papers_path)
    raw_data = load_data(settings.raw_papers_path)

    logger.info("Loaded %d raw papers", len(raw_data))
    logger.debug("Raw dataset shape: %s", raw_data.shape)
    logger.debug("Raw dataset columns: %s", raw_data.columns.tolist())

    # Process the raw dataset.
    logger.info("Preprocessing raw dataset")
    processed_data = preprocess_dataset(raw_data)

    logger.info(
        "Preprocessing completed: %d rows reduced to %d rows",
        len(raw_data),
        len(processed_data),
    )
    logger.debug("Processed dataset shape: %s", processed_data.shape)
    logger.debug(
        "Processed dataset columns: %s",
        processed_data.columns.tolist(),
    )

    # Save the dataset to disk in CSV format.
    logger.info(
        "Saving processed dataset to %s",
        settings.processed_papers_path,
    )
    save_to_csv(processed_data, settings.processed_papers_path)

    logger.info(
        "Preprocessing pipeline completed successfully in %.2f seconds",
        perf_counter() - start_time,
    )

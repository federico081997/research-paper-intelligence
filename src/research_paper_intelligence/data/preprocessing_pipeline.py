"""Coordinate the research-paper processing pipeline."""

from research_paper_intelligence.config import get_settings
from research_paper_intelligence.data.data_loader import load_data, save_to_csv
from research_paper_intelligence.data.preprocessing import preprocess_dataset


def run_preprocessing_pipeline() -> None:
    """Run the research-paper preprocessing pipeline."""
    settings = get_settings()

    # Load the raw dataset.
    raw_data = load_data(settings.raw_papers_path)

    # Process the raw dataset.
    processed_data = preprocess_dataset(raw_data)

    # Save the dataset to disk in CSV format.
    save_to_csv(processed_data, settings.processed_papers_path)

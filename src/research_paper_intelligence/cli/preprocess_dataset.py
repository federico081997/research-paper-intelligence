"""Main script for running the research-paper processing pipeline."""

from research_paper_intelligence.config import get_settings
from research_paper_intelligence.data.preprocessing_pipeline import (
    run_preprocessing_pipeline,
)
from research_paper_intelligence.logging_config import configure_logging


def main() -> None:
    """Configure and run the research-paper preprocessing pipeline."""
    settings = get_settings()

    configure_logging(settings)

    run_preprocessing_pipeline(settings)


if __name__ == "__main__":
    main()

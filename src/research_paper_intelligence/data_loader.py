"""Utilities for loading the data from the data folder."""

from pathlib import Path

import pandas as pd

from research_paper_intelligence.config import get_settings

REQUIRED_COLUMNS: set[str] = {
    "title",
    "abstract",
    "category",
    "authors",
    "published_date",
}

SETTINGS = get_settings()


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load the dataset from the data folder.

    Args:
        path: Path to the CSV file. If None, the configured processed
        data path is used.

    Returns:
        A DataFrame containing the dataset.

    Raises:
        FileNotFoundError: If the dataset is not found.
    """
    if not path:
        path = SETTINGS.processed_papers_path

    # Ensure the dataset exists before loading.
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    return pd.read_csv(path)


def save_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Save the DataFrame to disk in CSV format. .

    Args:
        df: Input DataFrame.
        path: Path where to save the DataFrame.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

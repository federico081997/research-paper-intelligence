from pathlib import Path

import pandas as pd

from research_paper_intelligence.config import get_settings

REQUIRED_COLUMNS: frozenset[str] = frozenset(
    {
        "title",
        "abstract",
        "category",
        "authors",
        "combined_text",
        "published_date",
    }
)

SETTINGS = get_settings()


def load_processed_data(path: Path | None = None) -> pd.DataFrame:
    """
    Load the processed data from the data folder

    Args:
        path: Path to the processed CSV file. If None, the configured processed
        data path is used.

    Returns:
        A DataFrame containing the processed data.

    Raises:
        FileNotFoundError: If the processed data is not found.
        KeyError: If the required columns are not present in the
            processed data.
    """

    if not path:
        path = SETTINGS.processed_papers_path

    # Ensure the processed dataset exists before loading
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found at: {path}")

    # Load the processed dataset into a Pandas DataFrame
    df = pd.read_csv(path)

    # Check if there are any missing columns in the dataset
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise KeyError(
            "The following columns are not present in the processed data: "
            + ", ".join(missing_columns)
        )

    # Ensure the published date column is in datetime format
    df["published_date"] = pd.to_datetime(df["published_date"], errors="raise")

    return df

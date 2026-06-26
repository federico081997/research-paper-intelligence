"""Tests performed on the ```preprocess_dataset`` script."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from scripts import preprocess_dataset as script


def test_main_runs_preprocessing_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    sample_dataframe: pd.DataFrame,
) -> None:
    """Run the loading, preprocessing, and saving steps in order."""
    raw_df = sample_dataframe.copy(deep=True)

    processed_df = sample_dataframe.copy(deep=True)
    processed_df["combined_text"] = (
        processed_df["title"] + " " + processed_df["summary"]
    )

    settings = SimpleNamespace(
        raw_papers_path=Path("data/raw/papers.csv"),
        processed_papers_path=Path("data/processed/papers.csv"),
    )

    # Replaces the real functions with mock functions
    mock_get_settings = Mock(return_value=settings)
    mock_load_data = Mock(return_value=raw_df)
    mock_preprocess_dataset = Mock(return_value=processed_df)
    mock_save_to_csv = Mock()

    monkeypatch.setattr(script, "get_settings", mock_get_settings)
    monkeypatch.setattr(script, "load_data", mock_load_data)
    monkeypatch.setattr(
        script,
        "preprocess_dataset",
        mock_preprocess_dataset,
    )
    monkeypatch.setattr(script, "save_to_csv", mock_save_to_csv)

    # Run the main script
    script.main()

    # Check whether the settings and load_data functions were only called once.
    mock_get_settings.assert_called_once_with()
    mock_load_data.assert_called_once_with(settings.raw_papers_path)

    # Check that the DataFrame loaded by load_data was passed to preprocessing.
    preprocessing_argument = mock_preprocess_dataset.call_args.args[0]
    assert preprocessing_argument is raw_df

    # Check that the processed result was saved to the correct path.
    saved_df, saved_path = mock_save_to_csv.call_args.args
    assert saved_df is processed_df
    assert saved_path == settings.processed_papers_path

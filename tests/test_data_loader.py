"""
Tests performed on the ``data loader`` module.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from research_paper_intelligence import data_loader


@pytest.fixture
def valid_processed_csv(tmp_path: Path) -> Path:
    """
    Create a valid processed csv file path.
    """
    csv_path = tmp_path / "arxiv_cleaned.csv"

    pd.DataFrame(
        {
            "title": ["Paper title"],
            "abstract": ["Paper abstract"],
            "category": ["Category"],
            "authors": ["Author"],
            "combined_text": ["Paper title Paper abstract"],
            "published_date": ["2025-01-10"],
        }
    ).to_csv(csv_path, index=False)

    return csv_path


class TestLoadProcessedData:
    """
    Tests performed on the ``load_processed_data`` function.
    """

    def test_loads_valid_processed_csv(
        self,
        valid_processed_csv: Path,
    ) -> None:
        """
        Tests that the CSV file is loaded correctly.
        """

        result = data_loader.load_processed_data(valid_processed_csv)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.loc[0, "title"] == "Paper title"

    def test_uses_supplied_path(
        self,
        valid_processed_csv: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Use the explicitly supplied path instead of configuration
        """

        # Configure a different path that does not exist
        monkeypatch.setattr(
            data_loader.SETTINGS,
            "processed_papers_path",
            tmp_path / "configured_missing.csv",
        )

        result = data_loader.load_processed_data(valid_processed_csv)

        assert result.loc[0, "title"] == "Paper title"

    def test_uses_configured_path(
        self,
        valid_processed_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Uses the configured processed data path when path is None
        """
        monkeypatch.setattr(
            data_loader.SETTINGS, "processed_papers_path", valid_processed_csv
        )

        result = data_loader.load_processed_data()

        assert result.loc[0, "title"] == "Paper title"

    def test_convert_to_datetime(
        self,
        valid_processed_csv: Path,
    ) -> None:
        """
        Test that the correct datetime string is returned.
        """

        result = data_loader.load_processed_data(valid_processed_csv)

        assert isinstance(result.loc[0, "published_date"], datetime)

    def test_file_not_found(self, tmp_path: Path) -> None:
        """
        Tests that the correct exception is raised when the file isn't found.
        """
        missing_path = tmp_path / "missing.csv"

        with pytest.raises(
            FileNotFoundError,
            match=f"Processed data not found at: {missing_path}",
        ):
            data_loader.load_processed_data(missing_path)

    @pytest.mark.parametrize(
        "missing_column",
        [
            ["title"],
            ["abstract"],
            ["category"],
            ["authors"],
            ["combined_text"],
            ["published_date"],
        ],
    )
    def test_missing_columns(
        self,
        tmp_path: Path,
        valid_processed_csv: Path,
        missing_column: list[str],
    ) -> None:
        """
        Tests that the correct exception is raised when the columns are
        missing.
        """

        # Define a dataFrame with missing columns
        invalid_df = pd.read_csv(valid_processed_csv).drop(
            missing_column, axis=1
        )
        invalid_path = tmp_path / f"missing_{missing_column}.csv"
        invalid_df.to_csv(invalid_path, index=False)

        with pytest.raises(
            KeyError,
            match="The following columns are not present in the processed "
            "data: " + ", ".join(missing_column),
        ):
            data_loader.load_processed_data(invalid_path)

"""Tests performed on the "data loader" module."""

from pathlib import Path

import pandas as pd
import pytest

from research_paper_intelligence.data.data_loader import (
    SETTINGS,
    load_data,
    save_to_csv,
)

# -----------------------------------------------------------------------------
#   TestLoadData
# -----------------------------------------------------------------------------


class TestLoadData:
    """Tests performed on the ``load_data`` function."""

    def test_loads_valid_csv(
        self,
        valid_csv: Path,
    ) -> None:
        """Tests that the CSV file is loaded correctly."""
        result = load_data(valid_csv)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert result.loc[0, "title"] == "Paper title"

    def test_uses_supplied_path(
        self,
        valid_csv: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Use the explicitly supplied path instead of configuration."""
        monkeypatch.setattr(
            SETTINGS,
            "processed_papers_path",
            tmp_path / "configured_missing.csv",
        )

        result = load_data(valid_csv)

        assert result.loc[0, "title"] == "Paper title"

    def test_uses_configured_path(
        self,
        valid_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Uses the configured processed data path when path is None."""
        monkeypatch.setattr(SETTINGS, "processed_papers_path", valid_csv)

        result = load_data()

        assert result.loc[0, "title"] == "Paper title"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raise an exception when the file is not found."""
        missing_path = tmp_path / "missing.csv"

        with pytest.raises(
            FileNotFoundError,
            match=f"Dataset not found at: {missing_path}",
        ):
            load_data(missing_path)


# -----------------------------------------------------------------------------
#   TestSaveToCsv
# -----------------------------------------------------------------------------


class TestSaveToCsv:
    """Tests for the save_to_csv function."""

    def test_saves_dataframe_to_csv(
        self,
        tmp_path: Path,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Save a DataFrame to the supplied CSV path."""
        output_path = tmp_path / "processed_papers.csv"

        save_to_csv(
            df=sample_dataframe,
            path=output_path,
        )

        assert output_path.exists()
        assert output_path.is_file()

        saved_dataframe = pd.read_csv(output_path)

        pd.testing.assert_frame_equal(
            saved_dataframe,
            sample_dataframe,
        )

    def test_creates_missing_parent_directories(
        self,
        tmp_path: Path,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Create missing parent directories before saving."""
        output_path = tmp_path / "data" / "processed" / "processed_papers.csv"

        assert not output_path.parent.exists()

        save_to_csv(
            df=sample_dataframe,
            path=output_path,
        )

        assert output_path.parent.exists()
        assert output_path.exists()

    def test_does_not_save_dataframe_index(
        self,
        tmp_path: Path,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Exclude the DataFrame index from the saved CSV."""
        output_path = tmp_path / "processed_papers.csv"

        save_to_csv(
            df=sample_dataframe,
            path=output_path,
        )

        saved_dataframe = pd.read_csv(output_path)

        assert "Unnamed: 0" not in saved_dataframe.columns
        assert list(saved_dataframe.columns) == list(sample_dataframe.columns)

    def test_overwrites_existing_csv(
        self,
        tmp_path: Path,
        sample_dataframe: pd.DataFrame,
    ) -> None:
        """Replace an existing CSV file with the new DataFrame."""
        output_path = tmp_path / "processed_papers.csv"

        pd.DataFrame(
            {
                "old_column": ["old value"],
            }
        ).to_csv(output_path, index=False)

        save_to_csv(
            df=sample_dataframe,
            path=output_path,
        )

        saved_dataframe = pd.read_csv(output_path)

        pd.testing.assert_frame_equal(
            saved_dataframe,
            sample_dataframe,
        )

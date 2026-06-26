"""Tests performed on the ``data loader`` module."""

from pathlib import Path

import pandas as pd
import pytest

from research_paper_intelligence import data_loader

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
        result = data_loader.load_data(valid_csv)

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
        # Configure a different path that does not exist
        monkeypatch.setattr(
            data_loader.SETTINGS,
            "processed_papers_path",
            tmp_path / "configured_missing.csv",
        )

        result = data_loader.load_data(valid_csv)

        assert result.loc[0, "title"] == "Paper title"

    def test_uses_configured_path(
        self,
        valid_csv: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Uses the configured processed data path when path is None."""
        monkeypatch.setattr(
            data_loader.SETTINGS, "processed_papers_path", valid_csv
        )

        result = data_loader.load_data()

        assert result.loc[0, "title"] == "Paper title"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raise an exception when the file is not found."""
        missing_path = tmp_path / "missing.csv"

        with pytest.raises(
            FileNotFoundError,
            match=f"Dataset not found at: {missing_path}",
        ):
            data_loader.load_data(missing_path)

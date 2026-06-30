"""Defines the class responsible for accessing the processed paper data."""

import pandas as pd

from research_paper_intelligence.domain.paper import Paper


class PaperRepository:
    """Class responsible for accessing the processed paper data."""

    def __init__(self, dataframe: pd.DataFrame):
        """Initializes the PaperRepository with the given dataframe."""
        self._dataframe = dataframe.reset_index(drop=True)

    def get_by_position(self, position: int) -> Paper:
        """Returns the paper at the given position.

        Args:
            position: The position of the paper to return.
        """
        row = self._dataframe.iloc[position]

        return Paper(
            paper_id=str(row["id"]),
            title=str(row["title"]),
            abstract=str(row["summary"]),
            authors=str(row["authors"]),
            category=str(row["category"]),
            published_date=pd.to_datetime(
                row["published_date"], errors="raise"
            ).date(),
        )

    def get_by_id(self, paper_id: str) -> Paper | None:
        """Returns the paper with the given id.

        Args:
            paper_id: The id of the paper to return.
        """
        match = self._dataframe[self._dataframe["id"] == paper_id]

        if match.empty:
            return None

        position = match.index[0]
        return self.get_by_position(position)

    def get_all(self) -> list[Paper]:
        """Returns every paper in dataset order."""
        return [
            self.get_by_position(position)
            for position in range(len(self._dataframe))
        ]

"""Tests the model validator within the configuration settings."""

import pytest
from pydantic import ValidationError

from research_paper_intelligence.config import Settings


class TestValidateScoreWeights:
    """Tests the validation of score weights."""

    def test_accepts_weights_that_sum_to_one(
        self,
        simple_settings: Settings,
    ) -> None:
        """Tests that weights that sum to one are accepted."""
        settings_data = simple_settings.model_dump()
        settings_data.update(
            {
                "semantic_weight": 0.6,
                "tfidf_weight": 0.2,
                "keyword_weight": 0.1,
                "recency_weight": 0.1,
            }
        )

        settings = Settings.model_validate(settings_data)

        assert settings.semantic_weight == 0.6

    def test_rejects_weights_that_do_not_sum_to_one(
        self,
        simple_settings: Settings,
    ) -> None:
        """Tests that it rejects weights that do not sum to one."""
        settings_data = simple_settings.model_dump()
        settings_data.update(
            {
                "semantic_weight": 0.6,
                "tfidf_weight": 0.3,
                "keyword_weight": 0.2,
                "recency_weight": 0.1,
            }
        )

        with pytest.raises(
            ValidationError,
            match="Score weights must sum to 1.0.",
        ):
            Settings.model_validate(settings_data)

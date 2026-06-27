"""Tests for application logging configuration."""

import logging
from unittest.mock import Mock

import pytest

from research_paper_intelligence import logging_config
from research_paper_intelligence.config import Settings


class TestConfigureLogging:
    """Tests for the ``configure_logging`` function."""

    @staticmethod
    @pytest.mark.parametrize(
        ("environment", "log_level", "expected_format"),
        [
            (
                "production",
                "INFO",
                "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            ),
            (
                "development",
                "DEBUG",
                "%(levelname)-8s | %(name)s | %(message)s",
            ),
            (
                "development",
                "INFO",
                "%(levelname)-8s | %(message)s",
            ),
        ],
    )
    def test_configure_logging(
        monkeypatch: pytest.MonkeyPatch,
        environment: str,
        log_level: str,
        expected_format: str,
    ) -> None:
        """Configure the expected level and format."""
        settings = Settings.model_construct(
            environment=environment,
            log_level=log_level,
        )
        mock_basic_config = Mock()
        mock_logger = Mock()

        monkeypatch.setattr(
            logging_config.logging,
            "basicConfig",
            mock_basic_config,
        )
        monkeypatch.setattr(logging_config, "logger", mock_logger)

        logging_config.configure_logging(settings)

        config = mock_basic_config.call_args.kwargs

        assert config["level"] == getattr(logging, log_level)
        assert config["format"] == expected_format
        assert config["datefmt"] == "%Y-%m-%d %H:%M:%S"
        assert config["force"] is True

        mock_logger.debug.assert_called_once_with(
            "Logging configured: environment=%s, level=%s",
            environment,
            log_level,
        )

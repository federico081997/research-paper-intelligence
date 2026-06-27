"""Configure application-wide logging."""

import logging
import sys

from research_paper_intelligence.config import Settings

logger = logging.getLogger(__name__)


def configure_logging(settings: Settings) -> None:
    """Configure the root logger using the application settings.

    Args:
        settings: Validated application configuration.
    """
    log_level = getattr(logging, settings.log_level)

    # Use a concise format during development and testing.
    if settings.environment == "production":
        # Detailed operational logs.
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    elif settings.log_level == "DEBUG":
        # Show the module name when debugging.
        log_format = "%(levelname)-8s | %(name)s | %(message)s"
    else:
        # Concise output for normal CLI usage.
        log_format = "%(levelname)-8s | %(message)s"

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    logger.debug(
        "Logging configured: environment=%s, level=%s",
        settings.environment,
        settings.log_level,
    )

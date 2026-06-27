"""Select the compute device used by machine-learning models."""

import logging
from functools import cache

import torch

from research_paper_intelligence.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


@cache
def get_device(
    preference: str = "auto",
) -> torch.device:
    """Select the requested PyTorch compute device.

    Args:
        preference: Preferred device. ``auto`` selects CUDA when available
            and otherwise falls back to the CPU.

    Returns:
        The selected PyTorch device.

    Raises:
        RuntimeError: If CUDA is requested but unavailable.
    """
    if preference == "cpu":
        logger.info("Using CPU")
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no CUDA-compatible GPU is available."
            )

        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    if torch.cuda.is_available():
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
        return torch.device("cuda")

    logger.info("CUDA is unavailable; using CPU")
    return torch.device("cpu")

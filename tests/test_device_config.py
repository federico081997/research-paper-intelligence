"""Tests for compute-device selection."""

from collections.abc import Iterator
from unittest.mock import Mock

import pytest
import torch

from research_paper_intelligence import device_config


class TestGetDevice:
    """Tests for the ``get_device`` function."""

    @pytest.fixture(autouse=True)
    def clear_device_cache(self) -> Iterator[None]:
        """Clear cached device selections between tests."""
        device_config.get_device.cache_clear()

        yield

        device_config.get_device.cache_clear()

    @staticmethod
    def test_selects_cpu(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Select the CPU when explicitly requested."""
        mock_logger = Mock()
        monkeypatch.setattr(device_config, "logger", mock_logger)

        result = device_config.get_device("cpu")

        assert result == torch.device("cpu")
        mock_logger.info.assert_called_once_with("Using CPU")

    @staticmethod
    def test_raises_when_cuda_is_unavailable(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Raise an error when unavailable CUDA is requested."""
        monkeypatch.setattr(
            device_config.torch.cuda,
            "is_available",
            lambda: False,
        )

        with pytest.raises(
            RuntimeError,
            match="CUDA was requested, but no CUDA-compatible GPU "
            "is available.",
        ):
            device_config.get_device("cuda")

    @staticmethod
    def test_selects_requested_cuda(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Select CUDA when explicitly requested and available."""
        mock_logger = Mock()

        monkeypatch.setattr(
            device_config.torch.cuda,
            "is_available",
            lambda: True,
        )
        monkeypatch.setattr(
            device_config.torch.cuda,
            "get_device_name",
            lambda _: "Test GPU",
        )
        monkeypatch.setattr(device_config, "logger", mock_logger)

        result = device_config.get_device("cuda")

        assert result == torch.device("cuda")
        mock_logger.info.assert_called_once_with(
            "Using GPU: %s",
            "Test GPU",
        )

    @staticmethod
    def test_auto_selects_cuda_when_available(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Automatically select CUDA when it is available."""
        monkeypatch.setattr(
            device_config.torch.cuda,
            "is_available",
            lambda: True,
        )
        monkeypatch.setattr(
            device_config.torch.cuda,
            "get_device_name",
            lambda _: "Test GPU",
        )

        result = device_config.get_device("auto")

        assert result == torch.device("cuda")

    @staticmethod
    def test_auto_falls_back_to_cpu(
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Fall back to the CPU when CUDA is unavailable."""
        mock_logger = Mock()

        monkeypatch.setattr(
            device_config.torch.cuda,
            "is_available",
            lambda: False,
        )
        monkeypatch.setattr(device_config, "logger", mock_logger)

        result = device_config.get_device("auto")

        assert result == torch.device("cpu")
        mock_logger.info.assert_called_once_with(
            "CUDA is unavailable; using CPU"
        )

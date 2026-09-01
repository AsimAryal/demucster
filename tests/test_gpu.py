"""Tests for production GPU configuration and device selection."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from pydantic import ValidationError

from demucster.core.config import Settings
from demucster.core.gpu import get_device


def test_require_cuda_fails_fast_when_unavailable() -> None:
    """Production mode must not silently run a CUDA workload on CPU."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        pytest.raises(RuntimeError, match="CUDA is unavailable"),
    ):
        get_device("cuda", require_cuda=True)


def test_cuda_index_is_honoured() -> None:
    """An explicit CUDA index is validated and returned."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.device_count", return_value=2),
        patch("torch.cuda.get_device_name", return_value="Mock GB10"),
        patch("torch.cuda.get_device_capability", return_value=(12, 1)),
    ):
        device = get_device("cuda:1", require_cuda=True)
    assert str(device) == "cuda:1"


def test_invalid_device_setting_is_rejected() -> None:
    """Unknown device aliases must not silently select CPU."""
    with pytest.raises(ValidationError, match="DEVICE"):
        Settings(device="gpu")


def test_invalid_segment_setting_is_rejected() -> None:
    """Non-numeric segment values fail during settings validation."""
    with pytest.raises(ValidationError):
        Settings(segment_size="default")

"""GPU device initialisation and memory introspection.

On NVIDIA DGX Spark (GB10 Blackwell) the 128 GB LPDDR5x pool is *unified*
between the Grace CPU and Blackwell GPU.  We therefore:

* Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` to prevent
  memory-fragmentation OOM on the large shared pool.
* Do **not** set ``CUDA_MANAGED_FORCE_DEVICE_ALLOC`` — the Grace Blackwell
  coherent driver manages residency far better when left alone.
* Use ``torch.cuda`` introspection instead of ``pynvml`` / ``nvidia-smi``
  because NVML may misreport memory statistics on GB10.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allocator tuning — set *before* any CUDA allocation
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def get_device(preferred: str = "cuda") -> torch.device:
    """Return a ``torch.device``, falling back to CPU when CUDA is unavailable.

    Parameters
    ----------
    preferred:
        Device string requested by the user / config (``"cuda"`` or ``"cpu"``).

    Returns
    -------
    torch.device
        The resolved device.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        dev = torch.device("cuda", 0)
        name = torch.cuda.get_device_name(dev)
        cap = torch.cuda.get_device_capability(dev)
        logger.info(
            "CUDA device selected: %s  (compute capability %d.%d)",
            name,
            *cap,
        )
        return dev

    if preferred == "cuda":
        logger.warning("CUDA requested but unavailable — falling back to CPU")
    return torch.device("cpu")


def get_gpu_info() -> dict[str, Any]:
    """Return a snapshot of GPU state using ``torch.cuda`` APIs.

    Keys returned:
        ``cuda_available``, ``device_name``, ``compute_capability``,
        ``vram_allocated_mb``, ``vram_reserved_mb``, ``vram_total_mb``.
    """
    if not torch.cuda.is_available():
        return {"cuda_available": False}

    idx = 0
    props = torch.cuda.get_device_properties(idx)
    return {
        "cuda_available": True,
        "device_name": props.name,
        "compute_capability": torch.cuda.get_device_capability(idx),
        "vram_allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024**2), 1),
        "vram_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024**2), 1),
        "vram_total_mb": round(props.total_mem / (1024**2), 1),
    }

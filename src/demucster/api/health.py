"""Health-check and system introspection endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter

from demucster.core.gpu import get_gpu_info
from demucster.models.schemas import HealthResponse, ModelInfo
from demucster.services.separator import SUPPORTED_MODELS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return system health including CUDA / GPU status and queue depth.

    Uses ``torch.cuda`` introspection (not pynvml) to avoid NVML
    misreporting on the DGX Spark's GB10 unified memory architecture.
    """
    from demucster.main import job_queue

    gpu = get_gpu_info()

    return HealthResponse(
        status="healthy",
        cuda_available=gpu["cuda_available"],
        device_name=gpu.get("device_name"),
        compute_capability=gpu.get("compute_capability"),
        vram_allocated_mb=gpu.get("vram_allocated_mb"),
        vram_total_mb=gpu.get("vram_total_mb"),
        active_jobs=job_queue.active_count if job_queue else 0,
        queued_jobs=job_queue.queued_count if job_queue else 0,
    )


@router.get("/models", response_model=list[ModelInfo])
async def list_models() -> list[ModelInfo]:
    """List all supported Demucs models with metadata."""
    return [
        ModelInfo(
            name=name,
            description=info["description"],
            stems=info["stems"],
            is_default=info.get("is_default", False),
        )
        for name, info in SUPPORTED_MODELS.items()
    ]

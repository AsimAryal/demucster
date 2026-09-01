"""Pydantic v2 request / response schemas for the Demucster API.

Every public model is a ``BaseModel`` (no dataclasses) to keep serialisation,
validation, and OpenAPI generation consistent across the codebase.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    """Lifecycle states for a separation job."""

    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class SeparationOptions(BaseModel):
    """User-configurable knobs for a separation job."""

    model: str = Field(
        default="htdemucs_ft",
        description=(
            "Demucs model name. Ranked best → fastest: "
            "htdemucs_ft, htdemucs, htdemucs_6s, mdx_extra, mdx_extra_q, mdx"
        ),
    )
    output_format: Literal["mp3", "wav"] = Field(
        default="mp3",
        description="Stem output format.",
    )
    mp3_bitrate: int = Field(
        default=320,
        ge=128,
        le=320,
        description="MP3 encoding bitrate (kbps). Ignored when output_format is wav.",
    )
    stems: list[str] | None = Field(
        default=None,
        description="Specific stems to extract (e.g. ['vocals', 'drums']). None = all.",
    )


class URLRequest(BaseModel):
    """Request body for URL-based job submission."""

    url: str = Field(
        ...,
        description="Public audio URL (YouTube, SoundCloud, direct link, etc.).",
    )
    options: SeparationOptions = Field(default_factory=SeparationOptions)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class JobResponse(BaseModel):
    """Public representation of a separation job returned by all job endpoints."""

    job_id: str
    status: JobStatus
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    filename: str | None = None
    stems: dict[str, str] | None = Field(
        default=None,
        description="Mapping of stem name → download URL path (available after COMPLETED).",
    )
    model: str
    output_format: str


class HealthResponse(BaseModel):
    """Response from the ``/health`` endpoint."""

    status: str = "healthy"
    cuda_available: bool = False
    device_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    vram_allocated_mb: float | None = None
    vram_total_mb: float | None = None
    active_jobs: int = 0
    queued_jobs: int = 0


class ModelInfo(BaseModel):
    """Metadata about a supported Demucs model."""

    name: str
    description: str
    stems: list[str]
    is_default: bool = False

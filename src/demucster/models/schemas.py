"""Pydantic v2 request / response schemas for the Demucster API.

Every public model is a ``BaseModel`` (no dataclasses) to keep serialisation,
validation, and OpenAPI generation consistent across the codebase.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic resolves this at runtime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class JobStatus(StrEnum):
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

    model: str | None = Field(
        default=None,
        description=(
            "Demucs model name. Ranked best → fastest: "
            "htdemucs_ft, htdemucs, htdemucs_6s, mdx_extra, mdx_extra_q, mdx"
        ),
    )
    output_format: Literal["mp3", "wav"] | None = Field(
        default=None,
        description="Stem output format.",
    )
    mp3_bitrate: int | None = Field(
        default=None,
        ge=128,
        le=320,
        description="MP3 encoding bitrate (kbps). Ignored when output_format is wav.",
    )
    stems: list[str] | None = Field(
        default=None,
        description="Specific stems to extract (e.g. ['vocals', 'drums']). None = all.",
    )

    @field_validator("model")
    @classmethod
    def _validate_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from demucster.services.separator import SUPPORTED_MODELS

        if value not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{value}'")
        return value

    @field_validator("stems")
    @classmethod
    def _validate_stem_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = list(dict.fromkeys(stem.strip().lower() for stem in value if stem.strip()))
        if not cleaned:
            raise ValueError("At least one stem must be requested")
        if any(not stem.isidentifier() for stem in cleaned):
            raise ValueError("Stem names may contain only letters, numbers, and underscores")
        return cleaned

    @model_validator(mode="after")
    def _validate_stems_for_model(self) -> SeparationOptions:
        if self.model is None or self.stems is None:
            return self
        from demucster.services.separator import SUPPORTED_MODELS

        available = set(SUPPORTED_MODELS[self.model]["stems"])
        invalid = sorted(set(self.stems) - available)
        if invalid:
            raise ValueError(
                f"Stems {invalid} are unavailable for model '{self.model}'; "
                f"choose from {sorted(available)}"
            )
        return self


class URLRequest(BaseModel):
    """Request body for URL-based job submission."""

    url: HttpUrl = Field(
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

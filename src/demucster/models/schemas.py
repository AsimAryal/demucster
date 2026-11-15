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
    exclude_stems: list[str] | None = Field(
        default=None,
        description=(
            "Stems to exclude — produces a mix of all other stems for each excluded "
            "stem (e.g. ['drums'] → a drumless track named 'no_drums'). "
            "Can be combined with 'stems' to get both individual stems and stemless mixes."
        ),
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

    @field_validator("stems", "exclude_stems")
    @classmethod
    def _validate_stem_names(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = list(dict.fromkeys(stem.strip().lower() for stem in value if stem.strip()))
        if not cleaned:
            raise ValueError("At least one stem must be specified")
        if any(not stem.isidentifier() for stem in cleaned):
            raise ValueError("Stem names may contain only letters, numbers, and underscores")
        return cleaned

    @model_validator(mode="after")
    def _validate_stems_for_model(self) -> SeparationOptions:
        if self.model is None:
            return self
        from demucster.services.separator import SUPPORTED_MODELS

        available = set(SUPPORTED_MODELS[self.model]["stems"])

        if self.stems is not None:
            invalid = sorted(set(self.stems) - available)
            if invalid:
                raise ValueError(
                    f"Stems {invalid} are unavailable for model '{self.model}'; "
                    f"choose from {sorted(available)}"
                )

        if self.exclude_stems is not None:
            invalid = sorted(set(self.exclude_stems) - available)
            if invalid:
                raise ValueError(
                    f"Exclude stems {invalid} are unavailable for model '{self.model}'; "
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
        description=(
            "Mapping of stem name → download URL path (available after COMPLETED). "
            "Stemless mixes use 'no_' prefix (e.g. 'no_drums')."
        ),
    )
    model: str
    output_format: str
    exclude_stems: list[str] | None = Field(
        default=None,
        description="Stems that were excluded to produce stemless mixes.",
    )


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
    lyrics_enabled: bool = False
    lyrics_active_jobs: int = 0
    lyrics_queued_jobs: int = 0
    ollama_configured: bool = False


class ModelInfo(BaseModel):
    """Metadata about a supported Demucs model."""

    name: str
    description: str
    stems: list[str]
    is_default: bool = False


# ---------------------------------------------------------------------------
# Lyrics extraction schemas
# ---------------------------------------------------------------------------


class LyricsWord(BaseModel):
    """Single word with start/end timestamps (seconds)."""

    word: str
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    score: float | None = Field(default=None, ge=0.0, le=1.0)


class LyricsLine(BaseModel):
    """Single karaoke display line composed of timed words."""

    index: int = Field(ge=0)
    start: float = Field(ge=0.0)
    end: float = Field(ge=0.0)
    text: str
    words: list[LyricsWord] = Field(default_factory=list)


class LyricsSummary(BaseModel):
    """Summary metadata attached to a completed lyrics extraction job."""

    language: str | None = None
    line_count: int = Field(default=0, ge=0)
    word_count: int = Field(default=0, ge=0)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    llm_refined: bool = False
    llm_model: str | None = None
    aligned_with_reference: bool = False


class LyricsDocument(BaseModel):
    """Canonical structured karaoke lyrics document serialized to ``lyrics.json``."""

    version: int = 1
    title: str | None = None
    model: str
    summary: LyricsSummary
    lines: list[LyricsLine] = Field(default_factory=list)


class LyricsOptions(BaseModel):
    """User-configurable options for a lyrics extraction job."""

    model: str | None = Field(
        default=None,
        description="Whisper model name (e.g. large-v3-turbo, large-v3, medium).",
    )
    language: str | None = Field(
        default=None,
        max_length=16,
        description="Optional ISO 639-1 language code (None = auto-detect).",
    )
    reference_lyrics: str | None = Field(
        default=None,
        max_length=50_000,
        description="Optional known lyrics text to force-align against the vocals.",
    )
    title: str | None = Field(
        default=None,
        max_length=255,
        description="Optional song title for metadata headers and LLM context.",
    )
    use_llm_cleanup: bool = Field(
        default=True,
        description="Use configured Ollama LLM to group karaoke lines and clean text.",
    )

    @field_validator("model")
    @classmethod
    def _validate_lyrics_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        from demucster.services.lyrics import SUPPORTED_LYRICS_MODELS

        if value not in SUPPORTED_LYRICS_MODELS:
            raise ValueError(f"Unsupported lyrics model '{value}'")
        return value

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if not cleaned or cleaned == "auto":
            return None
        if not cleaned.isalpha() or not 2 <= len(cleaned) <= 8:
            raise ValueError("Language must be an ISO code such as 'en' or 'es'")
        return cleaned

    @field_validator("reference_lyrics", "title")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class LyricsJobResponse(BaseModel):
    """Public representation of a lyrics extraction job."""

    job_id: str
    status: JobStatus
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    filename: str | None = None
    model: str
    artifacts: dict[str, str] | None = Field(
        default=None,
        description="Mapping of artifact kind ('json', 'lrc', 'ass', 'srt') → URL path.",
    )
    summary: LyricsSummary | None = None

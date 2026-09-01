"""Application configuration via Pydantic Settings.

All values are overridable through environment variables or a ``.env`` file
placed at the project root.  See ``.env.example`` for the full catalogue.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised, env-driven configuration for Demucster."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Model ---------------------------------------------------------------
    default_model: str = Field(
        default="htdemucs_ft",
        description=(
            "Demucs model loaded on startup. "
            "Options: htdemucs_ft, htdemucs, htdemucs_6s, mdx_extra, mdx_extra_q, mdx"
        ),
    )

    # --- Output --------------------------------------------------------------
    output_format: Literal["mp3", "wav"] = "mp3"
    mp3_bitrate: int = Field(default=320, ge=128, le=320)

    # --- GPU / Worker --------------------------------------------------------
    max_concurrent_jobs: int = Field(
        default=2,
        ge=1,
        description="Maximum jobs running on GPU simultaneously.",
    )
    device: str = Field(default="cuda", description="PyTorch device string.")
    segment_size: str | None = Field(
        default=None,
        description="Demucs segment length override (seconds). None = model default.",
    )

    # --- Storage -------------------------------------------------------------
    upload_dir: str = Field(default="/data/uploads")
    output_dir: str = Field(default="/data/outputs")
    max_file_size_mb: int = Field(default=500, ge=1)
    retention_hours: float = Field(
        default=24.0,
        ge=0.5,
        description="Hours before output files are purged.",
    )

    # --- Server --------------------------------------------------------------
    prewarm_model: bool = Field(
        default=True,
        description="Pre-load the default model into GPU memory on startup.",
    )
    log_level: str = Field(default="INFO")
    max_download_duration: int = Field(
        default=3600,
        ge=60,
        description="Maximum duration (seconds) for yt-dlp downloads.",
    )

    # --- Validators ----------------------------------------------------------
    @field_validator("default_model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        from demucster.services.separator import SUPPORTED_MODELS

        if v not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model '{v}'. Choose from: {', '.join(SUPPORTED_MODELS)}"
            )
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance.

    Using ``lru_cache`` ensures the ``.env`` file is read exactly once and
    subsequent calls return the same object without re-parsing.
    """
    return Settings()

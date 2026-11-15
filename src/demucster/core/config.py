"""Application configuration via Pydantic Settings.

All values are overridable through environment variables or a ``.env`` file
placed at the project root.  See ``.env.example`` for the full catalogue.
"""

from __future__ import annotations

import re
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
    require_cuda: bool = Field(
        default=False,
        description="Fail startup instead of falling back to CPU when CUDA was requested.",
    )
    segment_size: int | None = Field(
        default=None,
        ge=1,
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

    # --- Lyrics Extraction & Ollama ------------------------------------------
    lyrics_enabled: bool = Field(
        default=True,
        description="Enable timestamped lyrics extraction endpoints and worker.",
    )
    lyrics_dir: str = Field(default="/data/lyrics")
    whisper_cache_dir: str = Field(default="/data/model-cache/whisper")
    lyrics_default_model: str = Field(
        default="large-v3-turbo",
        description="Default Whisper model for lyrics extraction.",
    )
    lyrics_max_concurrent_jobs: int = Field(
        default=1,
        ge=1,
        description="Maximum lyrics jobs running concurrently.",
    )
    lyrics_prewarm_model: bool = Field(
        default=False,
        description="Pre-load the default Whisper model on startup.",
    )
    ollama_url: str | None = Field(
        default=None,
        description="Optional Ollama base URL for LLM line refinement (e.g. http://host.docker.internal:11434).",
    )
    ollama_model: str = Field(
        default="qwen3:27b",
        description="Ollama model name for karaoke line segmentation and cleanup.",
    )
    ollama_timeout_seconds: float = Field(
        default=120.0,
        ge=5.0,
        description="Request timeout (seconds) for Ollama refinement calls.",
    )

    # --- Validators ----------------------------------------------------------
    @field_validator("default_model")
    @classmethod
    def _validate_model(cls, v: str) -> str:
        from demucster.services.separator import SUPPORTED_MODELS

        if v not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{v}'. Choose from: {', '.join(SUPPORTED_MODELS)}")
        return v

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        value = value.lower().strip()
        if not re.fullmatch(r"(?:cpu|cuda(?::\d+)?)", value):
            raise ValueError("DEVICE must be 'cpu', 'cuda', or 'cuda:<index>'")
        return value

    @field_validator("lyrics_default_model")
    @classmethod
    def _validate_lyrics_model(cls, v: str) -> str:
        from demucster.services.lyrics import SUPPORTED_LYRICS_MODELS

        if v not in SUPPORTED_LYRICS_MODELS:
            raise ValueError(
                f"Unsupported lyrics model '{v}'. Choose from: {', '.join(SUPPORTED_LYRICS_MODELS)}"
            )
        return v

    @field_validator("ollama_url")
    @classmethod
    def _normalize_ollama_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().rstrip("/")
        return cleaned or None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton ``Settings`` instance.

    Using ``lru_cache`` ensures the ``.env`` file is read exactly once and
    subsequent calls return the same object without re-parsing.
    """
    return Settings()

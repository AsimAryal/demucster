"""Shared pytest fixtures for Demucster tests.

All fixtures mock GPU / Demucs dependencies so the test suite can run
on any CI runner without CUDA or ffmpeg.
"""

from __future__ import annotations

import wave
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_wav(path: Path, duration_secs: float = 1.0, channels: int = 2) -> Path:
    """Create a minimal valid WAV file filled with silence."""
    sample_rate = 44100
    num_frames = int(sample_rate * duration_secs)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * channels * num_frames)
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Provide temporary upload and output directories."""
    upload = tmp_path / "uploads"
    output = tmp_path / "outputs"
    lyrics = tmp_path / "lyrics"
    upload.mkdir()
    output.mkdir()
    lyrics.mkdir()
    return upload, output


@pytest.fixture()
def sample_wav(tmp_path: Path) -> Path:
    """Generate a 1-second stereo WAV file for upload tests."""
    return _generate_wav(tmp_path / "test_audio.wav")


@pytest.fixture()
def _mock_gpu():
    """Patch ``torch.cuda`` to report a fake CUDA device."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.get_device_name", return_value="Mock GPU"),
        patch("torch.cuda.get_device_capability", return_value=(12, 1)),
        patch("torch.cuda.empty_cache"),
    ):
        yield


@pytest.fixture()
def _mock_demucs_separator():
    """Replace ``DemucsService.separate`` with a fast mock that writes dummy stems."""

    def _fake_separate(
        self: Any,
        input_path: Path,
        output_dir: Path,
        model_name: str,
        output_format: str = "mp3",
        mp3_bitrate: int = 320,
        progress_callback: Any = None,
        requested_stems: list[str] | None = None,
        exclude_stems: list[str] | None = None,
    ) -> dict[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        stems = requested_stems or ["vocals", "drums", "bass", "other"]
        result: dict[str, Path] = {}
        for stem in stems:
            p = output_dir / f"{stem}.{output_format}"
            # Write a tiny valid-ish file (tests only check existence)
            p.write_bytes(b"\xff\xfb\x90\x00" * 25)  # 100 bytes
            result[stem] = p

        # Generate stemless (minus-one) mixes
        if exclude_stems:
            for excluded in exclude_stems:
                mix_name = f"no_{excluded}"
                p = output_dir / f"{mix_name}.{output_format}"
                p.write_bytes(b"\xff\xfb\x90\x00" * 25)  # 100 bytes
                result[mix_name] = p

        if progress_callback:
            progress_callback(50.0)
            progress_callback(100.0)
        return result

    with patch(
        "demucster.services.separator.DemucsService.separate",
        _fake_separate,
    ):
        yield


@pytest.fixture()
def _mock_demucs_warmup():
    """Disable model warmup so tests don't try to download real models."""
    with patch("demucster.services.separator.DemucsService.warmup"):
        yield


@pytest.fixture()
def _mock_lyrics_service():
    """Replace ``LyricsService.extract`` with a fast mock writing sample artifacts."""
    from demucster.models.schemas import LyricsWord
    from demucster.services.lyrics_formats import (
        build_lyrics_document,
        group_words_into_lines,
        write_lyrics_artifacts,
    )

    def _fake_extract(
        self: Any,
        *,
        input_path: Path,
        output_dir: Path,
        model_name: str,
        language: str | None = None,
        reference_lyrics: str | None = None,
        title: str | None = None,
        use_llm_cleanup: bool = True,
        ollama_url: str | None = None,
        ollama_model: str = "qwen3:27b",
        ollama_timeout_seconds: float = 120.0,
        progress_callback: Any = None,
    ) -> Any:
        words = [
            LyricsWord(word="Hello", start=0.2, end=0.6, score=0.95),
            LyricsWord(word="karaoke", start=0.65, end=1.1, score=0.92),
            LyricsWord(word="world", start=1.15, end=1.6, score=0.98),
        ]
        lines = group_words_into_lines(words)
        doc = build_lyrics_document(
            lines=lines,
            model=model_name,
            title=title or input_path.stem,
            language=language or "en",
            duration_seconds=1.6,
            llm_refined=bool(use_llm_cleanup and ollama_url),
            llm_model=ollama_model if (use_llm_cleanup and ollama_url) else None,
            aligned_with_reference=bool(reference_lyrics),
        )
        artifacts = write_lyrics_artifacts(doc, output_dir)
        if progress_callback:
            progress_callback(50.0)
            progress_callback(100.0)
        return artifacts, doc.summary

    with (
        patch("demucster.services.lyrics.LyricsService.extract", _fake_extract),
        patch("demucster.services.lyrics.LyricsService.warmup"),
    ):
        yield


@pytest.fixture()
def client(
    tmp_dirs: tuple[Path, Path],
    tmp_path: Path,
    _mock_gpu: None,
    _mock_demucs_separator: None,
    _mock_demucs_warmup: None,
    _mock_lyrics_service: None,
) -> TestClient:  # type: ignore[type-arg]
    """A ``TestClient`` wired to a fully-mocked Demucster app.

    Demucs and GPU calls are mocked so the suite needs no CUDA or model weights.
    """
    upload_dir, output_dir = tmp_dirs
    lyrics_dir = tmp_path / "lyrics"

    with patch.dict(
        "os.environ",
        {
            "UPLOAD_DIR": str(upload_dir),
            "OUTPUT_DIR": str(output_dir),
            "LYRICS_DIR": str(lyrics_dir),
            "LYRICS_PREWARM_MODEL": "false",
            "PREWARM_MODEL": "true",
            "DEVICE": "cpu",
            "LOG_LEVEL": "WARNING",
        },
    ):
        # Reset the settings cache so the patched env vars take effect
        from demucster.core.config import get_settings

        get_settings.cache_clear()

        from demucster.main import create_app

        app = create_app()
        with TestClient(app) as tc:
            yield tc

        get_settings.cache_clear()

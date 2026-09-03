"""Unit tests for service modules (audio, downloader, cleanup, separator)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from demucster.services.audio import (
    ALLOWED_EXTENSIONS,
    InvalidAudioError,
    save_stem,
    validate_audio_file,
)
from demucster.services.cleanup import _purge_old_files

# =====================================================================
# Audio validation
# =====================================================================


class TestAudioValidation:
    """``validate_audio_file`` — format and integrity checks."""

    def test_rejects_missing_file(self, tmp_path: Path):
        with pytest.raises(InvalidAudioError, match="not found"):
            validate_audio_file(tmp_path / "nope.wav")

    def test_rejects_unsupported_extension(self, tmp_path: Path):
        txt = tmp_path / "file.txt"
        txt.write_text("hello")
        with pytest.raises(InvalidAudioError, match="Unsupported format"):
            validate_audio_file(txt)

    def test_rejects_zero_frame_file(self, tmp_path: Path):
        wav = tmp_path / "empty.wav"
        wav.write_bytes(b"\x00" * 44)  # minimal WAV header, no frames

        with pytest.raises(InvalidAudioError):
            validate_audio_file(wav)

    def test_accepts_valid_wav(self, tmp_path: Path):
        from tests.conftest import _generate_wav

        wav = _generate_wav(tmp_path / "good.wav")
        validate_audio_file(wav)

    def test_allowed_extensions_complete(self):
        expected = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        assert expected == ALLOWED_EXTENSIONS

    def test_saves_wav_with_soundfile(self, tmp_path: Path) -> None:
        import soundfile
        import torch

        output = tmp_path / "stem.wav"
        save_stem(torch.zeros((2, 800)), 8000, output, output_format="wav")

        info = soundfile.info(output)
        assert info.frames == 800
        assert info.channels == 2

    def test_mp3_reports_missing_ffmpeg(self, tmp_path: Path) -> None:
        import torch

        output = tmp_path / "stem.mp3"
        with (
            patch(
                "demucster.services.audio.subprocess.run",
                side_effect=FileNotFoundError,
            ),
            pytest.raises(RuntimeError, match="ffmpeg not found"),
        ):
            save_stem(torch.zeros((2, 800)), 8000, output, output_format="mp3")

        assert not output.with_suffix(".tmp.wav").exists()


# =====================================================================
# Cleanup / purge
# =====================================================================


class TestCleanup:
    """``_purge_old_files`` — retention enforcement."""

    def test_deletes_old_files(self, tmp_path: Path):
        old_file = tmp_path / "old.wav"
        old_file.write_bytes(b"\x00" * 10)

        # Backdate the file
        import os

        past = time.time() - 100_000
        os.utime(old_file, (past, past))

        deleted = _purge_old_files([tmp_path], retention_seconds=3600)
        assert deleted == 1
        assert not old_file.exists()

    def test_preserves_recent_files(self, tmp_path: Path):
        recent = tmp_path / "recent.wav"
        recent.write_bytes(b"\x00" * 10)

        deleted = _purge_old_files([tmp_path], retention_seconds=3600)
        assert deleted == 0
        assert recent.exists()

    def test_removes_empty_directories(self, tmp_path: Path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        old_file = sub / "old.mp3"
        old_file.write_bytes(b"\x00")

        import os

        past = time.time() - 100_000
        os.utime(old_file, (past, past))

        _purge_old_files([tmp_path], retention_seconds=3600)
        assert not sub.exists()

    def test_handles_nonexistent_directory(self):
        # Should not raise
        deleted = _purge_old_files([Path("/nonexistent/dir")], retention_seconds=1)
        assert deleted == 0

    def test_preserves_protected_job_files(self, tmp_path: Path):
        protected_dir = tmp_path / "active-job"
        protected_dir.mkdir()
        old_file = protected_dir / "input.wav"
        old_file.write_bytes(b"\x00")

        import os

        past = time.time() - 100_000
        os.utime(old_file, (past, past))

        deleted = _purge_old_files(
            [tmp_path],
            retention_seconds=3600,
            protected_job_ids={"active-job"},
        )
        assert deleted == 0
        assert old_file.exists()


# =====================================================================
# Downloader (mocked — no actual network calls)
# =====================================================================


class TestDownloader:
    """``download_audio`` with mocked yt-dlp."""

    def test_download_returns_path_and_metadata(self, tmp_path: Path):
        from demucster.services.downloader import download_audio

        # Create the file that yt-dlp would produce
        wav_file = tmp_path / "Test Song.wav"
        wav_file.write_bytes(b"\x00" * 100)

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.return_value = {
            "title": "Test Song",
            "duration": 180,
            "uploader": "Test Artist",
        }
        mock_ydl_instance.prepare_filename.return_value = str(tmp_path / "Test Song.webm")
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with (
            patch("yt_dlp.YoutubeDL", return_value=mock_ydl_instance),
            patch("yt_dlp.utils.match_filter_func", return_value=None),
        ):
            path, meta = download_audio("https://example.com/video", tmp_path)

        assert path.exists()
        assert meta["title"] == "Test Song"
        assert meta["duration"] == 180

    def test_download_raises_on_failure(self, tmp_path: Path):
        import yt_dlp

        from demucster.services.downloader import DownloadError, download_audio

        mock_ydl_instance = MagicMock()
        mock_ydl_instance.extract_info.side_effect = yt_dlp.utils.DownloadError("404 Not Found")
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        with (
            patch("yt_dlp.YoutubeDL", return_value=mock_ydl_instance),
            patch("yt_dlp.utils.match_filter_func", return_value=None),
            pytest.raises(DownloadError, match="Failed to download"),
        ):
            download_audio("https://example.com/bad", tmp_path)


# =====================================================================
# Separator (model metadata only — no real inference)
# =====================================================================


class TestSeparatorMetadata:
    """``SUPPORTED_MODELS`` catalogue."""

    def test_all_models_have_required_keys(self):
        from demucster.services.separator import SUPPORTED_MODELS

        for name, info in SUPPORTED_MODELS.items():
            assert "description" in info, f"{name} missing description"
            assert "stems" in info, f"{name} missing stems"
            assert len(info["stems"]) >= 4, f"{name} has fewer than 4 stems"

    def test_htdemucs_ft_is_default(self):
        from demucster.services.separator import SUPPORTED_MODELS

        assert SUPPORTED_MODELS["htdemucs_ft"].get("is_default") is True

    def test_htdemucs_6s_has_six_stems(self):
        from demucster.services.separator import SUPPORTED_MODELS

        stems = SUPPORTED_MODELS["htdemucs_6s"]["stems"]
        assert len(stems) == 6
        assert "guitar" in stems
        assert "piano" in stems


def test_demucs_progress_uses_real_callback_fields() -> None:
    """Progress calculation consumes Demucs 4 callback keys."""
    from demucster.services.separator import _demucs_progress_pct

    progress = _demucs_progress_pct(
        {
            "models": 4,
            "model_idx_in_bag": 1,
            "segment_offset": 500,
            "audio_length": 1000,
            "state": "end",
        }
    )
    assert progress == pytest.approx(37.5)


# =====================================================================
# Exclude stems validation (SeparationOptions)
# =====================================================================


class TestExcludeStemsValidation:
    """``SeparationOptions.exclude_stems`` — schema-level validation."""

    def test_valid_exclude_stems(self) -> None:
        from demucster.models.schemas import SeparationOptions

        opts = SeparationOptions(model="htdemucs_ft", exclude_stems=["drums", "vocals"])
        assert opts.exclude_stems == ["drums", "vocals"]

    def test_exclude_stems_normalised_to_lowercase(self) -> None:
        from demucster.models.schemas import SeparationOptions

        opts = SeparationOptions(model="htdemucs_ft", exclude_stems=["DRUMS", " Vocals "])
        assert opts.exclude_stems == ["drums", "vocals"]

    def test_exclude_stems_invalid_for_model(self) -> None:
        from demucster.models.schemas import SeparationOptions

        with pytest.raises(ValueError, match=r"Exclude stems.*guitar.*unavailable"):
            SeparationOptions(model="htdemucs_ft", exclude_stems=["guitar"])

    def test_exclude_stems_valid_for_6s_model(self) -> None:
        from demucster.models.schemas import SeparationOptions

        opts = SeparationOptions(model="htdemucs_6s", exclude_stems=["guitar", "piano"])
        assert opts.exclude_stems == ["guitar", "piano"]

    def test_exclude_stems_deduplication(self) -> None:
        from demucster.models.schemas import SeparationOptions

        opts = SeparationOptions(model="htdemucs_ft", exclude_stems=["drums", "drums", "bass"])
        assert opts.exclude_stems == ["drums", "bass"]

    def test_exclude_stems_with_stems_both_validated(self) -> None:
        from demucster.models.schemas import SeparationOptions

        opts = SeparationOptions(model="htdemucs_ft", stems=["vocals"], exclude_stems=["drums"])
        assert opts.stems == ["vocals"]
        assert opts.exclude_stems == ["drums"]

    def test_exclude_stems_empty_list_rejected(self) -> None:
        from demucster.models.schemas import SeparationOptions

        with pytest.raises(ValueError, match="At least one stem must be specified"):
            SeparationOptions(model="htdemucs_ft", exclude_stems=[""])

    def test_exclude_stems_none_is_allowed(self) -> None:
        from demucster.models.schemas import SeparationOptions

        opts = SeparationOptions(model="htdemucs_ft", exclude_stems=None)
        assert opts.exclude_stems is None

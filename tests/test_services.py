"""Unit tests for service modules (audio, downloader, cleanup, separator)."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from demucster.services.audio import (
    ALLOWED_EXTENSIONS,
    InvalidAudioError,
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

        mock_info = MagicMock()
        mock_info.num_frames = 0
        with patch("demucster.services.audio.torchaudio.info", return_value=mock_info):
            with pytest.raises(InvalidAudioError, match="zero frames"):
                validate_audio_file(wav)

    def test_accepts_valid_wav(self, tmp_path: Path):
        from tests.conftest import _generate_wav

        wav = _generate_wav(tmp_path / "good.wav")

        mock_info = MagicMock()
        mock_info.num_frames = 44100
        with patch("demucster.services.audio.torchaudio.info", return_value=mock_info):
            # Should not raise
            validate_audio_file(wav)

    def test_allowed_extensions_complete(self):
        expected = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
        assert ALLOWED_EXTENSIONS == expected


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
        mock_ydl_instance.prepare_filename.return_value = str(
            tmp_path / "Test Song.webm"
        )
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)

        with patch("demucster.services.downloader.yt_dlp") as mock_ytdlp:
            mock_ytdlp.YoutubeDL.return_value = mock_ydl_instance
            mock_ytdlp.utils.match_filter_func = MagicMock(return_value=None)

            path, meta = download_audio("https://example.com/video", tmp_path)

        assert path.exists()
        assert meta["title"] == "Test Song"
        assert meta["duration"] == 180

    def test_download_raises_on_failure(self, tmp_path: Path):
        from demucster.services.downloader import DownloadError, download_audio

        with patch("demucster.services.downloader.yt_dlp") as mock_ytdlp:
            import yt_dlp as _real_ytdlp

            mock_ytdlp.utils.DownloadError = _real_ytdlp.utils.DownloadError
            mock_ytdlp.utils.match_filter_func = MagicMock(return_value=None)

            mock_ydl_instance = MagicMock()
            mock_ydl_instance.extract_info.side_effect = _real_ytdlp.utils.DownloadError(
                "404 Not Found"
            )
            mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_ydl_instance.__exit__ = MagicMock(return_value=False)
            mock_ytdlp.YoutubeDL.return_value = mock_ydl_instance

            with pytest.raises(DownloadError, match="Failed to download"):
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

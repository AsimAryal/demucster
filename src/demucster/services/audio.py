"""Audio format validation, normalisation, and encoding utilities.

Stems are saved as WAV via ``torchaudio`` or as high-quality MP3 via
``ffmpeg`` (must be on ``$PATH``).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import torch
import torchaudio

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS: set[str] = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


class InvalidAudioError(Exception):
    """Raised when an audio file cannot be read or has an unsupported format."""


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_audio_file(path: Path) -> None:
    """Verify that *path* is a readable audio file in an allowed format.

    Raises
    ------
    InvalidAudioError
        If the file is missing, has a disallowed extension, or cannot be
        decoded by torchaudio.
    """
    if not path.exists():
        raise InvalidAudioError(f"File not found: {path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise InvalidAudioError(
            f"Unsupported format '{path.suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    try:
        info = torchaudio.info(str(path))
        if info.num_frames == 0:
            raise InvalidAudioError(f"Audio file has zero frames: {path.name}")
    except InvalidAudioError:
        raise
    except Exception as exc:
        raise InvalidAudioError(f"Cannot read audio file '{path.name}': {exc}") from exc


# ---------------------------------------------------------------------------
# Stem persistence
# ---------------------------------------------------------------------------

def save_stem(
    tensor: torch.Tensor,
    sample_rate: int,
    output_path: Path,
    output_format: str = "mp3",
    mp3_bitrate: int = 320,
) -> None:
    """Write a stem tensor to disk as WAV or MP3.

    Parameters
    ----------
    tensor:
        Audio tensor with shape ``(channels, samples)`` or
        ``(1, channels, samples)``.
    sample_rate:
        Sample rate in Hz.
    output_path:
        Destination file path (suffix is ignored — *output_format* decides).
    output_format:
        ``"wav"`` or ``"mp3"``.
    mp3_bitrate:
        Target bitrate for MP3 encoding (kbps).  Ignored for WAV.

    Raises
    ------
    RuntimeError
        If ffmpeg encoding fails.
    ValueError
        If *output_format* is not ``"wav"`` or ``"mp3"``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Normalise tensor to 2-D (channels, samples) on CPU
    t = tensor.detach().cpu()
    if t.dim() == 3:
        t = t.squeeze(0)
    if t.dim() == 1:
        t = t.unsqueeze(0)

    if output_format == "wav":
        torchaudio.save(str(output_path), t, sample_rate)

    elif output_format == "mp3":
        # torchaudio's MP3 backend is limited; we shell out to ffmpeg for
        # reliable high-bitrate encoding.
        tmp_wav = output_path.with_suffix(".tmp.wav")
        try:
            torchaudio.save(str(tmp_wav), t, sample_rate)
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(tmp_wav),
                    "-codec:a", "libmp3lame",
                    "-b:a", f"{mp3_bitrate}k",
                    "-loglevel", "error",
                    str(output_path),
                ],
                check=True,
                capture_output=True,
            )
            if result.stderr:
                logger.debug("ffmpeg stderr: %s", result.stderr.decode(errors="replace"))
        except FileNotFoundError as exc:
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg to enable MP3 encoding."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"ffmpeg MP3 encoding failed: {exc.stderr.decode(errors='replace')}"
            ) from exc
        finally:
            tmp_wav.unlink(missing_ok=True)

    else:
        raise ValueError(f"Unsupported output format: {output_format!r}")

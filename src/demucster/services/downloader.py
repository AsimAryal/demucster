"""YouTube / URL audio downloader powered by yt-dlp.

Downloads the best-quality audio stream and post-processes it to WAV so
that Demucs receives uncompressed input regardless of source format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Safety guard — reject extremely long media to prevent abuse / disk fill.
_DEFAULT_MAX_DURATION = 3600  # 1 hour


class DownloadError(Exception):
    """Raised when a URL download fails or violates constraints."""


def download_audio(
    url: str,
    output_dir: Path,
    *,
    max_duration: int = _DEFAULT_MAX_DURATION,
) -> tuple[Path, dict[str, Any]]:
    """Download audio from *url* via yt-dlp.

    Parameters
    ----------
    url:
        Public URL (YouTube, SoundCloud, direct link, …).
    output_dir:
        Directory to save the downloaded file.
    max_duration:
        Maximum allowed duration in seconds.

    Returns
    -------
    tuple[Path, dict]
        ``(downloaded_file_path, metadata_dict)``

    Raises
    ------
    DownloadError
        On network errors, format issues, or duration violations.
    """
    import yt_dlp

    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(title).80s.%(ext)s")

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",
            }
        ],
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 3,
    }

    # Duration filter — reject videos longer than max_duration
    if max_duration > 0:
        ydl_opts["match_filter"] = yt_dlp.utils.match_filter_func(f"duration <= {max_duration}")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info: dict[str, Any] | None = ydl.extract_info(url, download=True)
            if info is None:
                raise DownloadError(f"yt-dlp returned no info for URL: {url}")

            # Locate the downloaded file (post-processor changes extension)
            raw_filename = ydl.prepare_filename(info)
            downloaded = Path(raw_filename).with_suffix(".wav")

            if not downloaded.exists():
                # Fallback: pick the first WAV in the output directory
                candidates = sorted(output_dir.glob("*.wav"))
                if candidates:
                    downloaded = candidates[0]
                else:
                    raise DownloadError(f"Downloaded file not found.  Expected: {downloaded}")

            metadata: dict[str, Any] = {
                "title": info.get("title", "Unknown"),
                "duration": info.get("duration"),
                "source_url": url,
                "uploader": info.get("uploader"),
                "thumbnail": info.get("thumbnail"),
            }

            logger.info(
                "Downloaded: '%s'  (%.0fs)  → %s",
                metadata["title"],
                metadata.get("duration") or 0,
                downloaded.name,
            )
            return downloaded, metadata

    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(f"Failed to download from {url}: {exc}") from exc
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"Unexpected error downloading {url}: {exc}") from exc

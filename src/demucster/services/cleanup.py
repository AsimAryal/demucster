"""Background file-retention service.

Periodically walks the upload and output directories, deleting files
(and empty parent directories) older than the configured retention window
to prevent unbounded disk growth.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


async def run_cleanup_loop(
    directories: list[Path],
    retention_hours: float,
    interval_seconds: float = 1800.0,
) -> None:
    """Purge expired files on a recurring schedule.

    This coroutine runs **indefinitely** and is meant to be launched as a
    background ``asyncio.Task`` from the application lifespan handler.

    Parameters
    ----------
    directories:
        Directories to scan (typically ``[UPLOAD_DIR, OUTPUT_DIR]``).
    retention_hours:
        Files older than this are deleted.
    interval_seconds:
        Seconds between cleanup sweeps (default 30 min).
    """
    retention_secs = retention_hours * 3600
    logger.info(
        "Cleanup loop started  retention=%.1fh  interval=%.0fs  dirs=%s",
        retention_hours,
        interval_seconds,
        [str(d) for d in directories],
    )

    while True:
        try:
            await asyncio.sleep(interval_seconds)
            deleted = _purge_old_files(directories, retention_secs)
            if deleted > 0:
                logger.info("Cleanup sweep: deleted %d expired file(s)", deleted)
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
            raise
        except Exception:
            logger.exception("Error during cleanup sweep")


def _purge_old_files(directories: list[Path], retention_seconds: float) -> int:
    """Delete files older than *retention_seconds*.  Returns deletion count."""
    now = time.time()
    deleted = 0

    for directory in directories:
        if not directory.exists():
            continue

        # Delete expired files
        for item in directory.rglob("*"):
            if not item.is_file():
                continue
            try:
                age = now - item.stat().st_mtime
                if age > retention_seconds:
                    item.unlink()
                    deleted += 1
            except OSError:
                logger.debug("Could not delete: %s", item, exc_info=True)

        # Prune empty subdirectories (deepest first so parents become empty)
        for subdir in sorted(directory.rglob("*"), key=lambda p: -len(p.parts)):
            if subdir.is_dir():
                try:
                    subdir.rmdir()  # only succeeds if empty
                except OSError:
                    pass

    return deleted

"""Background file-retention service.

Periodically walks the upload and output directories, deleting files
(and empty parent directories) older than the configured retention window
to prevent unbounded disk growth.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import suppress
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

logger = logging.getLogger(__name__)


async def run_cleanup_loop(
    directories: list[Path],
    retention_hours: float,
    interval_seconds: float = 1800.0,
    protected_ids_provider: Callable[[], Awaitable[set[str]]] | None = None,
    registry_cleanup: Callable[[float], Awaitable[int]] | None = None,
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
    protected_ids_provider:
        Async callback returning job directory names that must be preserved.
    registry_cleanup:
        Async callback that evicts expired terminal job records and their files.
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
            protected_ids = (
                await protected_ids_provider() if protected_ids_provider is not None else set()
            )
            expired_jobs = (
                await registry_cleanup(retention_secs) if registry_cleanup is not None else 0
            )
            deleted = _purge_old_files(
                directories,
                retention_secs,
                protected_job_ids=protected_ids,
            )
            if deleted > 0:
                logger.info("Cleanup sweep: deleted %d expired file(s)", deleted)
            if expired_jobs > 0:
                logger.info("Cleanup sweep: evicted %d expired job record(s)", expired_jobs)
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
            raise
        except Exception:
            logger.exception("Error during cleanup sweep")


def _purge_old_files(
    directories: list[Path],
    retention_seconds: float,
    *,
    protected_job_ids: set[str] | None = None,
) -> int:
    """Delete files older than *retention_seconds*.  Returns deletion count."""
    now = time.time()
    deleted = 0
    protected_job_ids = protected_job_ids or set()

    for directory in directories:
        if not directory.exists():
            continue

        # Delete expired files
        for item in directory.rglob("*"):
            if not item.is_file():
                continue
            try:
                relative = item.relative_to(directory)
            except ValueError:
                continue
            if relative.parts and relative.parts[0] in protected_job_ids:
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
                    relative = subdir.relative_to(directory)
                except ValueError:
                    continue
                if relative.parts and relative.parts[0] in protected_job_ids:
                    continue
                with suppress(OSError):
                    subdir.rmdir()  # only succeeds if empty

    return deleted

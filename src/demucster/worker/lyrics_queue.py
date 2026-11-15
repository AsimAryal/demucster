"""Asynchronous job queue for timestamped lyrics extraction."""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import torch
from pydantic import BaseModel, Field

from demucster.models.schemas import JobStatus, LyricsSummary
from demucster.worker.queue import JobProcessingError

logger = logging.getLogger(__name__)


class LyricsJobRecord(BaseModel):
    """Internal, mutable representation of a lyrics extraction job."""

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    status: JobStatus = JobStatus.QUEUED
    progress_pct: float = 0.0
    error_message: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    completed_at: datetime | None = None

    # File paths
    filename: str | None = None
    input_path: str | None = None
    output_dir: str | None = None
    owns_input: bool = True
    artifact_paths: dict[str, str] = Field(default_factory=dict)

    # Options
    model: str = "large-v3-turbo"
    language: str | None = None
    reference_lyrics: str | None = None
    title: str | None = None
    use_llm_cleanup: bool = True

    # Output summary
    summary: LyricsSummary | None = None


LyricsProcessFn = Callable[
    [LyricsJobRecord, Callable[[float], None]],
    tuple[dict[str, Path], LyricsSummary],
]


def _delete_lyrics_files(job: LyricsJobRecord) -> None:
    """Delete output artifacts and owned input files for a lyrics job."""
    if job.output_dir:
        shutil.rmtree(Path(job.output_dir), ignore_errors=True)

    if job.owns_input and job.input_path:
        input_path = Path(job.input_path)
        parent = input_path.parent
        if parent.name == job.job_id:
            shutil.rmtree(parent, ignore_errors=True)
        else:
            input_path.unlink(missing_ok=True)


class LyricsQueue:
    """In-process async queue bounding concurrent lyrics extraction jobs."""

    def __init__(self, max_concurrent: int = 1) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._registry: dict[str, LyricsJobRecord] = {}
        self._lock = asyncio.Lock()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._process_fn: LyricsProcessFn | None = None
        self._max_concurrent = max_concurrent
        self._active_count = 0
        self._accepting = True
        logger.info("LyricsQueue initialised: max_concurrent=%d", max_concurrent)

    async def start(self, process_fn: LyricsProcessFn) -> None:
        """Start background worker tasks for lyrics jobs."""
        self._process_fn = process_fn
        self._accepting = True
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(), name=f"lyrics-worker-{index}")
            for index in range(self._max_concurrent)
        ]

    async def stop(self) -> None:
        """Stop accepting jobs, drain queued work, and stop workers."""
        self._accepting = False
        await self._queue.join()
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

    async def enqueue(self, job: LyricsJobRecord) -> str:
        """Add *job* to the queue and return its ``job_id``."""
        if not self._accepting:
            raise RuntimeError("Lyrics queue is not accepting new work")
        async with self._lock:
            self._registry[job.job_id] = job
        await self._queue.put(job.job_id)
        logger.info("Lyrics job enqueued: %s  (model=%s)", job.job_id, job.model)
        return job.job_id

    async def get_job(self, job_id: str) -> LyricsJobRecord | None:
        """Look up a lyrics job by ID."""
        async with self._lock:
            return self._registry.get(job_id)

    async def delete_job(self, job_id: str) -> bool:
        """Delete a non-processing lyrics job and remove its files."""
        async with self._lock:
            job = self._registry.get(job_id)
            if job is not None and job.status == JobStatus.PROCESSING:
                raise JobProcessingError("Processing lyrics jobs cannot be deleted")
            if job is not None:
                self._registry.pop(job_id)

        if job is None:
            return False

        await asyncio.to_thread(_delete_lyrics_files, job)
        return True

    async def protected_job_ids(self) -> set[str]:
        """Return active job IDs whose directories must be preserved."""
        async with self._lock:
            return {
                job.job_id
                for job in self._registry.values()
                if job.status in {JobStatus.QUEUED, JobStatus.PROCESSING}
            }

    async def cleanup_expired(self, retention_seconds: float) -> int:
        """Evict expired terminal lyrics jobs from memory and disk."""
        cutoff = datetime.now(UTC).timestamp() - retention_seconds
        async with self._lock:
            expired = [
                job
                for job in self._registry.values()
                if job.status in {JobStatus.COMPLETED, JobStatus.FAILED}
                and (job.completed_at or job.created_at).timestamp() < cutoff
            ]
            for job in expired:
                self._registry.pop(job.job_id, None)

        for job in expired:
            await asyncio.to_thread(_delete_lyrics_files, job)
        return len(expired)

    async def update_progress(self, job_id: str, progress_pct: float) -> None:
        """Thread-safe progress update called from the executor thread."""
        async with self._lock:
            if job_id in self._registry:
                self._registry[job_id].progress_pct = max(0.0, min(progress_pct, 100.0))

    @property
    def active_count(self) -> int:
        """Number of lyrics jobs currently running."""
        return self._active_count

    @property
    def queued_count(self) -> int:
        """Number of lyrics jobs waiting in queue."""
        return self._queue.qsize()

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Unexpected error in lyrics worker loop")
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        async with self._lock:
            job = self._registry.get(job_id)
            if job is None:
                return
            job.status = JobStatus.PROCESSING
            job.progress_pct = 0.0

        self._active_count += 1
        loop = asyncio.get_running_loop()

        def _progress_cb(pct: float) -> None:
            try:
                asyncio.run_coroutine_threadsafe(self.update_progress(job_id, pct), loop)
            except RuntimeError:
                logger.debug("Lyrics progress update dropped while event loop is closing")

        try:
            if self._process_fn is None:
                raise RuntimeError("No lyrics process function registered")

            artifact_paths, summary = await loop.run_in_executor(
                None,
                lambda: self._process_fn(job, _progress_cb),  # type: ignore[arg-type]
            )
            if not artifact_paths:
                raise RuntimeError("Lyrics extraction completed without producing artifacts")

            async with self._lock:
                job.status = JobStatus.COMPLETED
                job.progress_pct = 100.0
                job.completed_at = datetime.now(UTC)
                job.artifact_paths = {key: str(value) for key, value in artifact_paths.items()}
                job.summary = summary
            logger.info("Lyrics job completed: %s", job_id)

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            await asyncio.to_thread(_delete_lyrics_files, job)
            async with self._lock:
                job.status = JobStatus.FAILED
                job.error_message = (
                    "GPU out of memory during lyrics extraction. "
                    "Try a lighter Whisper model (e.g. medium)."
                )
                job.completed_at = datetime.now(UTC)
            logger.error("Lyrics job %s failed: CUDA OOM", job_id)

        except Exception as exc:
            await asyncio.to_thread(_delete_lyrics_files, job)
            async with self._lock:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at = datetime.now(UTC)
            logger.exception("Lyrics job %s failed", job_id)
        finally:
            self._active_count -= 1

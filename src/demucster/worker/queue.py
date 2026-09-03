"""Asynchronous job queue with bounded GPU concurrency.

The queue is entirely in-process — no Redis or Celery required. A fixed number
of worker tasks consume an ``asyncio.Queue`` in FIFO order, bounding concurrent
GPU inference on the DGX Spark's unified 128 GB memory pool.

Job state is tracked in a ``dict[str, JobRecord]`` guarded by an
``asyncio.Lock``.  This is suitable for a single-instance deployment; for
multi-replica horizontal scaling replace with Redis/SQL.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from demucster.models.schemas import JobStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job record — Pydantic model (not dataclass) per project convention
# ---------------------------------------------------------------------------


class JobRecord(BaseModel):
    """Internal, mutable representation of a separation job.

    Not returned directly over the API — see ``JobResponse`` in
    ``demucster.models.schemas`` for the public view.
    """

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
    stem_paths: dict[str, str] = Field(default_factory=dict)

    # Separation options
    model: str = "htdemucs_ft"
    output_format: str = "mp3"
    mp3_bitrate: int = 320
    requested_stems: list[str] | None = None
    exclude_stems: list[str] | None = None

    # URL download metadata
    source_url: str | None = None
    source_metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Process function type
# ---------------------------------------------------------------------------
ProcessFn = Callable[[JobRecord, Callable[[float], None]], dict[str, Path]]
"""Signature: (job, progress_callback) → {stem_name: output_path}"""


class JobProcessingError(RuntimeError):
    """Raised when an operation is unsafe while a job is processing."""


def _delete_job_files(job: JobRecord, *, include_input: bool = True) -> None:
    """Delete files associated with a job."""
    if job.output_dir:
        shutil.rmtree(Path(job.output_dir), ignore_errors=True)

    if include_input and job.input_path:
        input_path = Path(job.input_path)
        parent = input_path.parent
        if parent.name == job.job_id:
            shutil.rmtree(parent, ignore_errors=True)
        else:
            input_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------


class JobQueue:
    """In-process async job queue with bounded GPU concurrency.

    Attributes
    ----------
    active_count : int
        Number of jobs currently running on the GPU.
    queued_count : int
        Number of jobs waiting in the FIFO queue.
    """

    def __init__(self, max_concurrent: int = 2) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._registry: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._process_fn: ProcessFn | None = None
        self._max_concurrent = max_concurrent
        self._active_count = 0
        self._accepting = True
        logger.info("JobQueue initialised: max_concurrent=%d", max_concurrent)

    # -- Lifecycle -----------------------------------------------------------

    async def start(self, process_fn: ProcessFn) -> None:
        """Start the background worker loop.

        Parameters
        ----------
        process_fn:
            Synchronous callable ``(job, progress_cb) → stem_paths``.
            Executed inside ``loop.run_in_executor`` to avoid blocking
            the event loop.
        """
        self._process_fn = process_fn
        self._accepting = True
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(), name=f"job-worker-{index}")
            for index in range(self._max_concurrent)
        ]
        logger.info("Started %d job worker(s)", self._max_concurrent)

    async def stop(self) -> None:
        """Stop accepting jobs, drain queued work, and stop worker loops."""
        self._accepting = False
        await self._queue.join()
        for task in self._worker_tasks:
            task.cancel()
        if self._worker_tasks:
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()
        logger.info("Job workers stopped")

    # -- Public interface ----------------------------------------------------

    async def enqueue(self, job: JobRecord) -> str:
        """Append *job* to the queue and return its ``job_id``."""
        if not self._accepting:
            raise RuntimeError("Job queue is not accepting new work")
        async with self._lock:
            self._registry[job.job_id] = job
        await self._queue.put(job.job_id)
        logger.info("Job enqueued: %s  (model=%s)", job.job_id, job.model)
        return job.job_id

    async def get_job(self, job_id: str) -> JobRecord | None:
        """Look up a job by ID. Returns ``None`` if not found."""
        async with self._lock:
            return self._registry.get(job_id)

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[JobRecord]:
        """Return a page of jobs, optionally filtered by *status*.

        Results are sorted newest-first.
        """
        async with self._lock:
            jobs = list(self._registry.values())

        if status is not None:
            jobs = [j for j in jobs if j.status == status]

        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[offset : offset + limit]

    async def delete_job(self, job_id: str) -> bool:
        """Remove a job from the registry and delete its files on disk.

        Returns ``True`` if the job existed and was removed, ``False`` otherwise.
        """
        async with self._lock:
            job = self._registry.get(job_id)
            if job is not None and job.status == JobStatus.PROCESSING:
                raise JobProcessingError("Processing jobs cannot be deleted")
            if job is not None:
                self._registry.pop(job_id)

        if job is None:
            return False

        await asyncio.to_thread(_delete_job_files, job)
        logger.info("Job deleted: %s", job_id)
        return True

    async def protected_job_ids(self) -> set[str]:
        """Return IDs whose files must not be removed by retention cleanup."""
        async with self._lock:
            return {
                job.job_id
                for job in self._registry.values()
                if job.status in {JobStatus.QUEUED, JobStatus.PROCESSING}
            }

    async def cleanup_expired(self, retention_seconds: float) -> int:
        """Remove expired terminal jobs from memory and disk."""
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
            await asyncio.to_thread(_delete_job_files, job)
        return len(expired)

    async def update_progress(self, job_id: str, progress_pct: float) -> None:
        """Thread-safe progress update (called from the executor thread)."""
        async with self._lock:
            if job_id in self._registry:
                self._registry[job_id].progress_pct = max(0.0, min(progress_pct, 100.0))

    @property
    def active_count(self) -> int:
        """Number of jobs currently running."""
        return self._active_count

    @property
    def queued_count(self) -> int:
        """Number of jobs waiting in the FIFO queue."""
        return self._queue.qsize()

    # -- Internal worker -----------------------------------------------------

    async def _worker_loop(self) -> None:
        """Pull and execute jobs; one loop exists per concurrency slot."""
        while True:
            job_id = await self._queue.get()
            try:
                await self._run_job(job_id)
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                raise
            except Exception:
                logger.exception("Unexpected error in worker loop")
            finally:
                self._queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        """Execute one job in the default thread-pool executor."""
        async with self._lock:
            job = self._registry.get(job_id)
            if job is None:
                logger.debug("Skipping deleted queued job %s", job_id)
                return
            job.status = JobStatus.PROCESSING
            job.progress_pct = 0.0

        self._active_count += 1
        loop = asyncio.get_running_loop()

        def _progress_cb(pct: float) -> None:
            try:
                asyncio.run_coroutine_threadsafe(self.update_progress(job_id, pct), loop)
            except RuntimeError:
                logger.debug("Progress update dropped while event loop is closing")

        try:
            if self._process_fn is None:
                raise RuntimeError("No process function registered with the queue")

            stem_paths = await loop.run_in_executor(
                None,
                lambda: self._process_fn(job, _progress_cb),  # type: ignore[arg-type]
            )
            if not stem_paths:
                raise RuntimeError("Separation completed without producing stems")

            async with self._lock:
                job.status = JobStatus.COMPLETED
                job.progress_pct = 100.0
                job.completed_at = datetime.now(UTC)
                job.stem_paths = {key: str(value) for key, value in stem_paths.items()}
            logger.info("Job completed: %s  (%d stems)", job_id, len(stem_paths))

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            await asyncio.to_thread(_delete_job_files, job, include_input=False)
            async with self._lock:
                job.status = JobStatus.FAILED
                job.error_message = (
                    "GPU out of memory. Try a shorter audio file or a lighter model "
                    "(e.g. mdx_extra_q instead of htdemucs_ft)."
                )
                job.completed_at = datetime.now(UTC)
            logger.error("Job %s failed: CUDA OOM", job_id)

        except Exception as exc:
            await asyncio.to_thread(_delete_job_files, job, include_input=False)
            async with self._lock:
                job.status = JobStatus.FAILED
                job.error_message = str(exc)
                job.completed_at = datetime.now(UTC)
            logger.exception("Job %s failed", job_id)
        finally:
            self._active_count -= 1

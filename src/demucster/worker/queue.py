"""Asynchronous job queue with bounded GPU concurrency.

The queue is entirely in-process — no Redis or Celery required — using an
``asyncio.Queue`` for FIFO ordering and an ``asyncio.Semaphore`` to cap
concurrent GPU inference and prevent VRAM exhaustion on the DGX Spark's
unified 128 GB memory pool.

Job state is tracked in a ``dict[str, JobRecord]`` guarded by an
``asyncio.Lock``.  This is suitable for a single-instance deployment; for
multi-replica horizontal scaling replace with Redis/SQL.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from pydantic import BaseModel, Field

from demucster.models.schemas import JobStatus

if TYPE_CHECKING:
    pass

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
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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

    # URL download metadata
    source_url: str | None = None
    source_metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Process function type
# ---------------------------------------------------------------------------
ProcessFn = Callable[[JobRecord, Callable[[float], None]], dict[str, Path]]
"""Signature: (job, progress_callback) → {stem_name: output_path}"""


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
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._registry: dict[str, JobRecord] = {}
        self._lock = asyncio.Lock()
        self._worker_task: asyncio.Task[None] | None = None
        self._process_fn: ProcessFn | None = None
        self._max_concurrent = max_concurrent
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
        self._worker_task = asyncio.create_task(self._worker_loop(), name="job-worker")
        logger.info("Job worker loop started")

    async def stop(self) -> None:
        """Cancel the worker loop and wait for it to finish."""
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("Job worker loop stopped")

    # -- Public interface ----------------------------------------------------

    async def enqueue(self, job: JobRecord) -> str:
        """Append *job* to the queue and return its ``job_id``."""
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
            job = self._registry.pop(job_id, None)

        if job is None:
            return False

        # Clean output directory
        if job.output_dir:
            output_path = Path(job.output_dir)
            if output_path.exists():
                shutil.rmtree(output_path, ignore_errors=True)

        # Clean input file / upload directory
        if job.input_path:
            input_path = Path(job.input_path)
            parent = input_path.parent
            if parent.name == job.job_id and parent.exists():
                shutil.rmtree(parent, ignore_errors=True)
            elif input_path.exists():
                input_path.unlink(missing_ok=True)

        logger.info("Job deleted: %s", job_id)
        return True

    async def update_progress(self, job_id: str, progress_pct: float) -> None:
        """Thread-safe progress update (called from the executor thread)."""
        async with self._lock:
            if job_id in self._registry:
                self._registry[job_id].progress_pct = min(progress_pct, 100.0)

    @property
    def active_count(self) -> int:
        """Number of jobs currently holding the GPU semaphore."""
        return self._max_concurrent - self._semaphore._value  # noqa: SLF001

    @property
    def queued_count(self) -> int:
        """Number of jobs waiting in the FIFO queue."""
        return self._queue.qsize()

    # -- Internal worker -----------------------------------------------------

    async def _worker_loop(self) -> None:
        """Pull job IDs from the queue and spawn bounded tasks."""
        while True:
            try:
                job_id = await self._queue.get()
                asyncio.create_task(self._run_job(job_id), name=f"job-{job_id[:8]}")
            except asyncio.CancelledError:
                logger.info("Worker loop cancelled")
                raise
            except Exception:
                logger.exception("Unexpected error in worker loop")

    async def _run_job(self, job_id: str) -> None:
        """Acquire the GPU semaphore and execute one job."""
        async with self._semaphore:
            job = await self.get_job(job_id)
            if job is None:
                logger.warning("Job %s vanished before processing", job_id)
                return

            # Mark as PROCESSING
            async with self._lock:
                job.status = JobStatus.PROCESSING
                job.progress_pct = 0.0

            # Capture the running loop so the sync callback can schedule updates
            loop = asyncio.get_running_loop()

            def _progress_cb(pct: float) -> None:
                asyncio.run_coroutine_threadsafe(
                    self.update_progress(job_id, pct),
                    loop,
                )

            try:
                if self._process_fn is None:
                    raise RuntimeError("No process function registered with the queue")

                stem_paths: dict[str, Path] = await loop.run_in_executor(
                    None,
                    lambda: self._process_fn(job, _progress_cb),  # type: ignore[arg-type]
                )

                async with self._lock:
                    job.status = JobStatus.COMPLETED
                    job.progress_pct = 100.0
                    job.completed_at = datetime.now(timezone.utc)
                    job.stem_paths = {k: str(v) for k, v in stem_paths.items()}

                logger.info("Job completed: %s  (%d stems)", job_id, len(stem_paths))

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                async with self._lock:
                    job.status = JobStatus.FAILED
                    job.error_message = (
                        "GPU out of memory. Try a shorter audio file or a lighter model "
                        "(e.g. mdx_extra_q instead of htdemucs_ft)."
                    )
                    job.completed_at = datetime.now(timezone.utc)
                logger.error("Job %s failed: CUDA OOM", job_id)

            except Exception as exc:
                async with self._lock:
                    job.status = JobStatus.FAILED
                    job.error_message = str(exc)
                    job.completed_at = datetime.now(timezone.utc)
                logger.exception("Job %s failed", job_id)

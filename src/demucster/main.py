"""FastAPI application factory with async lifespan management.

The lifespan handler wires up:
* GPU device selection
* Demucs model pre-warming
* Job queue worker loop
* Background file-retention cleanup task
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from demucster.api.router import api_router
from demucster.core.config import get_settings
from demucster.core.gpu import get_device
from demucster.core.logging import setup_logging
from demucster.services.cleanup import run_cleanup_loop
from demucster.services.separator import DemucsService
from demucster.worker.queue import JobQueue, JobRecord

logger = logging.getLogger(__name__)

# Module-level references so endpoint code can import them directly.
job_queue: JobQueue | None = None
demucs_service: DemucsService | None = None


# ---------------------------------------------------------------------------
# Synchronous job processor (runs inside ``loop.run_in_executor``)
# ---------------------------------------------------------------------------

def _process_job(
    job: JobRecord,
    progress_callback: callable,  # type: ignore[valid-type]
) -> dict[str, Path]:
    """Execute one separation job.

    Called from a thread-pool executor by the ``JobQueue`` worker.
    If the job was submitted via URL, the audio is downloaded first.
    """
    assert demucs_service is not None

    settings = get_settings()
    input_path: Path | None = Path(job.input_path) if job.input_path else None
    output_dir: Path = Path(job.output_dir) if job.output_dir else Path(settings.output_dir) / job.job_id

    # --- URL jobs: download audio first ---
    if job.source_url and input_path is None:
        from demucster.services.downloader import download_audio

        upload_dir = Path(settings.upload_dir) / job.job_id
        downloaded, metadata = download_audio(
            job.source_url,
            upload_dir,
            max_duration=settings.max_download_duration,
        )
        input_path = downloaded
        job.input_path = str(input_path)
        job.filename = downloaded.name
        job.source_metadata = metadata

    if input_path is None:
        raise RuntimeError("Job has no input_path and no source_url")

    return demucs_service.separate(
        input_path=input_path,
        output_dir=output_dir,
        model_name=job.model,
        output_format=job.output_format,
        mp3_bitrate=job.mp3_bitrate,
        progress_callback=progress_callback,
        requested_stems=job.requested_stems,
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle."""
    global job_queue, demucs_service  # noqa: PLW0603

    settings = get_settings()
    setup_logging(level=settings.log_level)

    logger.info("Starting Demucster v0.1.0")

    # Create data directories
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)

    # GPU device
    device = get_device(preferred=settings.device)
    logger.info("Device: %s", device)

    # Demucs service
    segment = int(settings.segment_size) if settings.segment_size else None
    demucs_service = DemucsService(device=device, segment=segment)

    if settings.prewarm_model:
        try:
            demucs_service.warmup(settings.default_model)
        except Exception:
            logger.warning(
                "Model pre-warm failed — first job will incur cold-start latency",
                exc_info=True,
            )

    # Job queue
    job_queue = JobQueue(max_concurrent=settings.max_concurrent_jobs)
    await job_queue.start(process_fn=_process_job)

    # Cleanup loop
    cleanup_task = asyncio.create_task(
        run_cleanup_loop(
            directories=[Path(settings.upload_dir), Path(settings.output_dir)],
            retention_hours=settings.retention_hours,
        ),
        name="cleanup",
    )

    logger.info("Demucster is ready")

    yield  # ---- application runs here ----

    # Shutdown
    logger.info("Shutting down Demucster …")
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    if job_queue is not None:
        await job_queue.stop()

    logger.info("Demucster shutdown complete")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the configured ``FastAPI`` application."""
    app = FastAPI(
        title="Demucster",
        description=(
            "GPU-accelerated audio stem separation microservice powered by "
            "Demucs.  Upload an audio file or provide a URL and get back "
            "individual stems (vocals, drums, bass, other, …)."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS — wide open for single-instance / internal deployments
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    return app


# The ``app`` object that Uvicorn imports: ``uvicorn demucster.main:app``
app = create_app()

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
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from demucster.api.router import api_router
from demucster.core.config import get_settings
from demucster.core.gpu import get_device
from demucster.core.logging import setup_logging
from demucster.services.cleanup import run_cleanup_loop
from demucster.services.lyrics import LyricsService
from demucster.services.separator import DemucsService
from demucster.worker.lyrics_queue import LyricsJobRecord, LyricsQueue
from demucster.worker.queue import JobQueue, JobRecord

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

    from demucster.models.schemas import LyricsSummary

logger = logging.getLogger(__name__)

# Module-level references so endpoint code can import them directly.
job_queue: JobQueue | None = None
demucs_service: DemucsService | None = None
lyrics_queue: LyricsQueue | None = None
lyrics_service: LyricsService | None = None


# ---------------------------------------------------------------------------
# Synchronous job processor (runs inside ``loop.run_in_executor``)
# ---------------------------------------------------------------------------


def _process_job(
    job: JobRecord,
    progress_callback: Callable[[float], None],
) -> dict[str, Path]:
    """Execute one separation job.

    Called from a thread-pool executor by the ``JobQueue`` worker.
    If the job was submitted via URL, the audio is downloaded first.
    """
    assert demucs_service is not None

    settings = get_settings()
    input_path: Path | None = Path(job.input_path) if job.input_path else None
    output_dir = Path(job.output_dir) if job.output_dir else Path(settings.output_dir) / job.job_id

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

    from demucster.services.audio import validate_audio_file

    validate_audio_file(input_path)

    return demucs_service.separate(
        input_path=input_path,
        output_dir=output_dir,
        model_name=job.model,
        output_format=job.output_format,
        mp3_bitrate=job.mp3_bitrate,
        progress_callback=progress_callback,
        requested_stems=job.requested_stems,
        exclude_stems=job.exclude_stems,
    )


def _process_lyrics_job(
    job: LyricsJobRecord,
    progress_callback: Callable[[float], None],
) -> tuple[dict[str, Path], LyricsSummary]:
    """Execute one timestamped lyrics extraction job in the executor pool."""
    assert lyrics_service is not None

    if not job.input_path:
        raise RuntimeError("Lyrics job has no input_path")

    settings = get_settings()
    input_path = Path(job.input_path)
    output_dir = Path(job.output_dir) if job.output_dir else Path(settings.lyrics_dir) / job.job_id

    if job.owns_input:
        from demucster.services.audio import validate_audio_file

        validate_audio_file(input_path)
    elif not input_path.exists():
        raise RuntimeError(f"Stem audio not found: {input_path}")

    return lyrics_service.extract(
        input_path=input_path,
        output_dir=output_dir,
        model_name=job.model,
        language=job.language,
        reference_lyrics=job.reference_lyrics,
        title=job.title,
        use_llm_cleanup=job.use_llm_cleanup,
        ollama_url=settings.ollama_url,
        ollama_model=settings.ollama_model,
        ollama_timeout_seconds=settings.ollama_timeout_seconds,
        progress_callback=progress_callback,
    )


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown lifecycle."""
    global job_queue, demucs_service, lyrics_queue, lyrics_service

    settings = get_settings()
    setup_logging(level=settings.log_level)

    logger.info("Starting Demucster v0.1.0")

    # Create data directories
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    if settings.lyrics_enabled:
        Path(settings.lyrics_dir).mkdir(parents=True, exist_ok=True)

    # GPU device
    device = get_device(preferred=settings.device, require_cuda=settings.require_cuda)
    logger.info("Device: %s", device)

    # Demucs service
    demucs_service = DemucsService(device=device, segment=settings.segment_size)

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

    # Lyrics service & queue
    if settings.lyrics_enabled:
        lyrics_service = LyricsService(
            device=device,
            cache_dir=Path(settings.whisper_cache_dir),
        )
        if settings.lyrics_prewarm_model:
            try:
                lyrics_service.warmup(settings.lyrics_default_model)
            except Exception:
                logger.warning(
                    "Lyrics model pre-warm failed — first lyrics job will incur cold-start",
                    exc_info=True,
                )
        lyrics_queue = LyricsQueue(max_concurrent=settings.lyrics_max_concurrent_jobs)
        await lyrics_queue.start(process_fn=_process_lyrics_job)
    else:
        lyrics_service = None
        lyrics_queue = None

    async def _protected_ids() -> set[str]:
        ids = await job_queue.protected_job_ids()
        if lyrics_queue is not None:
            ids |= await lyrics_queue.protected_job_ids()
        return ids

    async def _cleanup_expired(retention_seconds: float) -> int:
        removed = await job_queue.cleanup_expired(retention_seconds)
        if lyrics_queue is not None:
            removed += await lyrics_queue.cleanup_expired(retention_seconds)
        return removed

    cleanup_dirs = [Path(settings.upload_dir), Path(settings.output_dir)]
    if settings.lyrics_enabled:
        cleanup_dirs.append(Path(settings.lyrics_dir))

    # Cleanup loop
    cleanup_task = asyncio.create_task(
        run_cleanup_loop(
            directories=cleanup_dirs,
            retention_hours=settings.retention_hours,
            protected_ids_provider=_protected_ids,
            registry_cleanup=_cleanup_expired,
        ),
        name="cleanup",
    )

    logger.info("Demucster is ready")

    yield  # ---- application runs here ----

    # Shutdown
    logger.info("Shutting down Demucster …")
    cleanup_task.cancel()
    with suppress(asyncio.CancelledError):
        await cleanup_task

    if lyrics_queue is not None:
        await lyrics_queue.stop()

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
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    return app


# The ``app`` object that Uvicorn imports: ``uvicorn demucster.main:app``
app = create_app()

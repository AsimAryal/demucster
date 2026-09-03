"""Job management endpoints — upload, URL submission, status, download."""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError
from starlette.background import BackgroundTask

from demucster.core.config import get_settings
from demucster.models.schemas import (
    JobResponse,
    JobStatus,
    SeparationOptions,
    URLRequest,
)
from demucster.services.audio import ALLOWED_EXTENSIONS, InvalidAudioError, validate_audio_file
from demucster.worker.queue import JobProcessingError, JobQueue, JobRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _queue() -> JobQueue:
    """Retrieve the global ``JobQueue`` set during app lifespan."""
    from demucster.main import job_queue  # deferred to avoid circular imports

    if job_queue is None:
        raise HTTPException(503, "Service not ready — job queue uninitialised")
    return job_queue


def _resolve_options(
    *,
    model: str | None,
    output_format: str | None,
    mp3_bitrate: int | None,
    stems: list[str] | None,
    exclude_stems: list[str] | None = None,
) -> SeparationOptions:
    """Apply configured defaults and validate a complete option set."""
    settings = get_settings()
    try:
        options = SeparationOptions(
            model=model or settings.default_model,
            output_format=output_format or settings.output_format,
            mp3_bitrate=mp3_bitrate if mp3_bitrate is not None else settings.mp3_bitrate,
            stems=stems,
            exclude_stems=exclude_stems,
        )
    except ValidationError as exc:
        raise HTTPException(
            422,
            detail=exc.errors(include_url=False, include_context=False),
        ) from exc
    return options


def _remove_file(path: Path) -> None:
    """Remove a temporary response file."""
    path.unlink(missing_ok=True)


def _job_to_response(job: JobRecord) -> JobResponse:
    """Project an internal ``JobRecord`` to the public ``JobResponse`` schema."""
    stems: dict[str, str] | None = None
    if job.stem_paths:
        stems = {name: f"/api/v1/jobs/{job.job_id}/stems/{name}" for name in job.stem_paths}
    return JobResponse(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
        filename=job.filename,
        stems=stems,
        model=job.model,
        output_format=job.output_format,
        exclude_stems=job.exclude_stems,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/upload", status_code=202, response_model=JobResponse)
async def upload_file(
    file: Annotated[UploadFile, File(description="Audio file to separate")],
    model: Annotated[str | None, Form()] = None,
    output_format: Annotated[str | None, Form()] = None,
    mp3_bitrate: Annotated[int | None, Form()] = None,
    stems: Annotated[
        str | None,
        Form(description="Comma-separated stem names to extract (e.g. 'vocals,drums')"),
    ] = None,
    exclude_stems: Annotated[
        str | None,
        Form(
            description=(
                "Comma-separated stem names to exclude — produces stemless mixes "
                "(e.g. 'drums' → a drumless track named 'no_drums')"
            ),
        ),
    ] = None,
) -> JobResponse:
    """Upload an audio file for stem separation.

    Accepts ``multipart/form-data`` with the file and optional separation
    parameters.  Returns ``202 Accepted`` with a ``JobResponse`` whose
    ``status`` is ``QUEUED``.
    """
    settings = get_settings()
    queue = _queue()

    # --- Validate filename & extension ---
    if not file.filename:
        raise HTTPException(400, "Filename is required")

    normalised_name = file.filename.replace("\\", "/")
    safe_name = Path(normalised_name).name
    if safe_name != normalised_name or safe_name in {"", ".", ".."}:
        raise HTTPException(422, "Filename must not contain a path")

    suffix = Path(safe_name).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            422,
            f"Unsupported format '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    parsed_stems = [stem.strip() for stem in stems.split(",")] if stems else None
    parsed_exclude = [stem.strip() for stem in exclude_stems.split(",")] if exclude_stems else None
    options = _resolve_options(
        model=model,
        output_format=output_format,
        mp3_bitrate=mp3_bitrate,
        stems=parsed_stems,
        exclude_stems=parsed_exclude,
    )
    assert options.model is not None
    assert options.output_format is not None
    assert options.mp3_bitrate is not None

    # --- Build job record ---
    job = JobRecord(
        model=options.model,
        output_format=options.output_format,
        mp3_bitrate=options.mp3_bitrate,
        filename=safe_name,
        requested_stems=options.stems,
        exclude_stems=options.exclude_stems,
    )

    # --- Stream upload to disk with a hard size limit ---
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    upload_dir = Path(settings.upload_dir) / job.job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / safe_name
    total_bytes = 0
    try:
        with input_path.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        413,
                        f"File exceeds maximum size of {settings.max_file_size_mb} MB",
                    )
                output.write(chunk)
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    # --- Validate audio integrity ---
    try:
        validate_audio_file(input_path)
    except InvalidAudioError as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(422, str(exc)) from exc

    job.input_path = str(input_path)
    job.output_dir = str(Path(settings.output_dir) / job.job_id)

    try:
        await queue.enqueue(job)
    except RuntimeError as exc:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(503, "Service is shutting down") from exc
    logger.info("Upload accepted  job=%s  file=%s", job.job_id, safe_name)
    return _job_to_response(job)


@router.post("/url", status_code=202, response_model=JobResponse)
async def submit_url(request: URLRequest) -> JobResponse:
    """Submit a URL (YouTube, SoundCloud, …) for download + separation.

    The audio is fetched by the worker as part of the queued job — the
    endpoint returns immediately with ``202 Accepted``.
    """
    settings = get_settings()
    queue = _queue()
    options = _resolve_options(
        model=request.options.model,
        output_format=request.options.output_format,
        mp3_bitrate=request.options.mp3_bitrate,
        stems=request.options.stems,
        exclude_stems=request.options.exclude_stems,
    )
    assert options.model is not None
    assert options.output_format is not None
    assert options.mp3_bitrate is not None

    job = JobRecord(
        model=options.model,
        output_format=options.output_format,
        mp3_bitrate=options.mp3_bitrate,
        source_url=str(request.url),
        requested_stems=options.stems,
        exclude_stems=options.exclude_stems,
    )
    job.output_dir = str(Path(settings.output_dir) / job.job_id)

    try:
        await queue.enqueue(job)
    except RuntimeError as exc:
        raise HTTPException(503, "Service is shutting down") from exc
    logger.info("URL job accepted  job=%s  url=%s", job.job_id, request.url)
    return _job_to_response(job)


@router.get("", response_model=list[JobResponse])
async def list_jobs(
    status: Annotated[JobStatus | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[JobResponse]:
    """List all jobs, optionally filtered by status.  Newest first."""
    queue = _queue()
    jobs = await queue.list_jobs(status=status, limit=limit, offset=offset)
    return [_job_to_response(j) for j in jobs]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Retrieve status, progress, and metadata for a single job."""
    queue = _queue()
    job = await queue.get_job(job_id)
    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    return _job_to_response(job)


@router.get("/{job_id}/stems/{stem_name}")
async def download_stem(job_id: str, stem_name: str) -> FileResponse:
    """Stream / download an individual separated stem audio file."""
    queue = _queue()
    job = await queue.get_job(job_id)

    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job is not completed (status: {job.status.value})")
    if stem_name not in job.stem_paths:
        raise HTTPException(
            404,
            f"Stem '{stem_name}' not found. Available: {list(job.stem_paths.keys())}",
        )

    stem_path = Path(job.stem_paths[stem_name])
    if not stem_path.exists():
        raise HTTPException(410, "Stem file has been cleaned up — re-submit the job")

    media_type = "audio/mpeg" if stem_path.suffix == ".mp3" else "audio/wav"
    return FileResponse(
        path=stem_path,
        media_type=media_type,
        filename=f"{stem_name}{stem_path.suffix}",
    )


@router.get("/{job_id}/source")
async def download_source(job_id: str) -> FileResponse:
    """Stream the full source mix retained for a completed job."""
    queue = _queue()
    job = await queue.get_job(job_id)

    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job is not completed (status: {job.status.value})")
    if not job.input_path:
        raise HTTPException(404, "No source audio is available for this job")

    source_path = Path(job.input_path)
    if not source_path.exists():
        raise HTTPException(410, "Source audio has been cleaned up — re-submit the job")

    media_types = {
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".mp3": "audio/mpeg",
        ".ogg": "audio/ogg",
        ".wav": "audio/wav",
    }
    return FileResponse(
        path=source_path,
        media_type=media_types.get(source_path.suffix.lower(), "application/octet-stream"),
        filename=job.filename or source_path.name,
    )


@router.get("/{job_id}/download-zip")
async def download_zip(job_id: str) -> FileResponse:
    """Bundle all stems into a ZIP archive and stream it to the client."""
    queue = _queue()
    job = await queue.get_job(job_id)

    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job is not completed (status: {job.status.value})")
    if not job.stem_paths:
        raise HTTPException(404, "No stems available for this job")

    stem_paths = [Path(path) for path in job.stem_paths.values()]
    if any(not path.exists() for path in stem_paths):
        raise HTTPException(410, "One or more stem files have been cleaned up")

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as temporary:
        zip_path = Path(temporary.name)
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for stem_path in stem_paths:
                archive.write(stem_path, stem_path.name)
    except Exception:
        zip_path.unlink(missing_ok=True)
        raise

    base_name = Path(job.filename).stem if job.filename else job.job_id
    zip_filename = f"{base_name}_stems.zip"

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=zip_filename,
        background=BackgroundTask(_remove_file, zip_path),
    )


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    """Delete a job and purge its associated files from disk."""
    queue = _queue()
    try:
        deleted = await queue.delete_job(job_id)
    except JobProcessingError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, f"Job not found: {job_id}")

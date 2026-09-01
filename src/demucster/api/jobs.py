"""Job management endpoints — upload, URL submission, status, download."""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from demucster.core.config import get_settings
from demucster.models.schemas import JobResponse, JobStatus, URLRequest
from demucster.services.audio import ALLOWED_EXTENSIONS, InvalidAudioError, validate_audio_file
from demucster.worker.queue import JobRecord

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _queue():
    """Retrieve the global ``JobQueue`` set during app lifespan."""
    from demucster.main import job_queue  # deferred to avoid circular imports

    if job_queue is None:
        raise HTTPException(503, "Service not ready — job queue uninitialised")
    return job_queue


def _job_to_response(job: JobRecord) -> JobResponse:
    """Project an internal ``JobRecord`` to the public ``JobResponse`` schema."""
    stems: dict[str, str] | None = None
    if job.stem_paths:
        stems = {
            name: f"/api/v1/jobs/{job.job_id}/stems/{name}"
            for name in job.stem_paths
        }
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
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload", status_code=202, response_model=JobResponse)
async def upload_file(
    file: Annotated[UploadFile, File(description="Audio file to separate")],
    model: Annotated[str, Form()] = "htdemucs_ft",
    output_format: Annotated[str, Form()] = "mp3",
    mp3_bitrate: Annotated[int, Form()] = 320,
    stems: Annotated[
        str | None,
        Form(description="Comma-separated stem names to extract (e.g. 'vocals,drums')"),
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

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            422,
            f"Unsupported format '{suffix}'. "
            f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # --- Read & validate size ---
    content = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(413, f"File exceeds maximum size of {settings.max_file_size_mb} MB")

    # --- Build job record ---
    parsed_stems = [s.strip() for s in stems.split(",") if s.strip()] if stems else None
    job = JobRecord(
        model=model,
        output_format=output_format,
        mp3_bitrate=mp3_bitrate,
        filename=file.filename,
        requested_stems=parsed_stems,
    )

    # --- Persist upload ---
    upload_dir = Path(settings.upload_dir) / job.job_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    input_path = upload_dir / file.filename
    input_path.write_bytes(content)

    # --- Validate audio integrity ---
    try:
        validate_audio_file(input_path)
    except InvalidAudioError as exc:
        input_path.unlink(missing_ok=True)
        upload_dir.rmdir()
        raise HTTPException(422, str(exc)) from exc

    job.input_path = str(input_path)
    job.output_dir = str(Path(settings.output_dir) / job.job_id)

    await queue.enqueue(job)
    logger.info("Upload accepted  job=%s  file=%s", job.job_id, file.filename)
    return _job_to_response(job)


@router.post("/url", status_code=202, response_model=JobResponse)
async def submit_url(request: URLRequest) -> JobResponse:
    """Submit a URL (YouTube, SoundCloud, …) for download + separation.

    The audio is fetched by the worker as part of the queued job — the
    endpoint returns immediately with ``202 Accepted``.
    """
    settings = get_settings()
    queue = _queue()

    job = JobRecord(
        model=request.options.model,
        output_format=request.options.output_format,
        mp3_bitrate=request.options.mp3_bitrate,
        source_url=str(request.url),
        requested_stems=request.options.stems,
    )
    job.output_dir = str(Path(settings.output_dir) / job.job_id)

    await queue.enqueue(job)
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
            f"Stem '{stem_name}' not found. "
            f"Available: {list(job.stem_paths.keys())}",
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


@router.get("/{job_id}/download-zip")
async def download_zip(job_id: str) -> StreamingResponse:
    """Bundle all stems into a ZIP archive and stream it to the client."""
    queue = _queue()
    job = await queue.get_job(job_id)

    if job is None:
        raise HTTPException(404, f"Job not found: {job_id}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"Job is not completed (status: {job.status.value})")
    if not job.stem_paths:
        raise HTTPException(404, "No stems available for this job")

    # Build ZIP in memory
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for _stem_name, stem_path_str in job.stem_paths.items():
            stem_path = Path(stem_path_str)
            if stem_path.exists():
                zf.write(stem_path, stem_path.name)

    buf.seek(0)
    base_name = Path(job.filename).stem if job.filename else job.job_id
    zip_filename = f"{base_name}_stems.zip"

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{zip_filename}"'},
    )


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    """Delete a job and purge its associated files from disk."""
    queue = _queue()
    deleted = await queue.delete_job(job_id)
    if not deleted:
        raise HTTPException(404, f"Job not found: {job_id}")

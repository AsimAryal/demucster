"""Lyrics extraction endpoints — upload, from-job, status, artifact download."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import ValidationError

from demucster.core.config import get_settings
from demucster.models.schemas import (
    JobStatus,
    LyricsJobResponse,
    LyricsOptions,
)
from demucster.services.audio import ALLOWED_EXTENSIONS, InvalidAudioError, validate_audio_file
from demucster.worker.lyrics_queue import LyricsJobRecord, LyricsQueue
from demucster.worker.queue import JobProcessingError, JobQueue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/lyrics", tags=["lyrics"])

ARTIFACT_MEDIA_TYPES: dict[str, tuple[str, str]] = {
    "json": ("application/json", "lyrics.json"),
    "lrc": ("text/plain; charset=utf-8", "lyrics.lrc"),
    "ass": ("text/plain; charset=utf-8", "lyrics.ass"),
    "srt": ("text/plain; charset=utf-8", "lyrics.srt"),
}


def _lyrics_queue() -> LyricsQueue:
    """Retrieve the global ``LyricsQueue`` initialised during lifespan."""
    from demucster.main import lyrics_queue

    settings = get_settings()
    if not settings.lyrics_enabled:
        raise HTTPException(404, "Lyrics extraction is disabled")
    if lyrics_queue is None:
        raise HTTPException(503, "Service not ready — lyrics queue uninitialised")
    return lyrics_queue


def _sep_queue() -> JobQueue:
    """Retrieve the separation ``JobQueue`` for ``from-job`` submissions."""
    from demucster.main import job_queue

    if job_queue is None:
        raise HTTPException(503, "Service not ready — job queue uninitialised")
    return job_queue


def _resolve_lyrics_options(
    *,
    model: str | None,
    language: str | None,
    reference_lyrics: str | None,
    title: str | None,
    use_llm_cleanup: bool,
) -> LyricsOptions:
    """Apply configured defaults and validate lyrics extraction options."""
    settings = get_settings()
    try:
        return LyricsOptions(
            model=model or settings.lyrics_default_model,
            language=language,
            reference_lyrics=reference_lyrics,
            title=title,
            use_llm_cleanup=use_llm_cleanup,
        )
    except ValidationError as exc:
        raise HTTPException(
            422,
            detail=exc.errors(include_url=False, include_context=False),
        ) from exc


def _lyrics_to_response(job: LyricsJobRecord) -> LyricsJobResponse:
    """Project internal ``LyricsJobRecord`` to public ``LyricsJobResponse``."""
    artifacts: dict[str, str] | None = None
    if job.artifact_paths:
        artifacts = {
            kind: f"/api/v1/lyrics/{job.job_id}/artifacts/{kind}" for kind in job.artifact_paths
        }
    return LyricsJobResponse(
        job_id=job.job_id,
        status=job.status,
        progress_pct=job.progress_pct,
        error_message=job.error_message,
        created_at=job.created_at,
        completed_at=job.completed_at,
        filename=job.filename,
        model=job.model,
        artifacts=artifacts,
        summary=job.summary,
    )


@router.post("/upload", status_code=202, response_model=LyricsJobResponse)
async def upload_for_lyrics(
    file: Annotated[UploadFile, File(description="Vocal stem or audio file to transcribe")],
    model: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    reference_lyrics: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
    use_llm_cleanup: Annotated[bool, Form()] = True,
) -> LyricsJobResponse:
    """Upload an audio file (ideally isolated vocals) for timestamped lyrics extraction."""
    settings = get_settings()
    queue = _lyrics_queue()

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

    options = _resolve_lyrics_options(
        model=model,
        language=language,
        reference_lyrics=reference_lyrics,
        title=title,
        use_llm_cleanup=use_llm_cleanup,
    )
    assert options.model is not None

    job = LyricsJobRecord(
        filename=safe_name,
        model=options.model,
        language=options.language,
        reference_lyrics=options.reference_lyrics,
        title=options.title,
        use_llm_cleanup=options.use_llm_cleanup,
        owns_input=True,
    )

    max_bytes = settings.max_file_size_mb * 1024 * 1024
    lyrics_dir = Path(settings.lyrics_dir) / job.job_id
    upload_dir = lyrics_dir / "input"
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
        shutil.rmtree(lyrics_dir, ignore_errors=True)
        raise

    try:
        validate_audio_file(input_path)
    except InvalidAudioError as exc:
        shutil.rmtree(lyrics_dir, ignore_errors=True)
        raise HTTPException(422, str(exc)) from exc

    job.input_path = str(input_path)
    job.output_dir = str(lyrics_dir)

    try:
        await queue.enqueue(job)
    except RuntimeError as exc:
        shutil.rmtree(lyrics_dir, ignore_errors=True)
        raise HTTPException(503, "Service is shutting down") from exc

    logger.info("Lyrics upload accepted  job=%s  file=%s", job.job_id, safe_name)
    return _lyrics_to_response(job)


@router.post("/from-job/{sep_job_id}", status_code=202, response_model=LyricsJobResponse)
async def extract_from_separation_job(
    sep_job_id: str,
    body: LyricsOptions | None = None,
) -> LyricsJobResponse:
    """Run timestamped lyrics extraction on the ``vocals`` stem of a completed job."""
    settings = get_settings()
    queue = _lyrics_queue()
    sep_queue = _sep_queue()

    sep_job = await sep_queue.get_job(sep_job_id)
    if sep_job is None:
        raise HTTPException(404, f"Separation job not found: {sep_job_id}")
    if sep_job.status != JobStatus.COMPLETED:
        raise HTTPException(
            409, f"Separation job is not completed (status: {sep_job.status.value})"
        )

    vocals_raw = sep_job.stem_paths.get("vocals")
    if not vocals_raw:
        raise HTTPException(404, "Separation job does not have a 'vocals' stem")

    vocals_path = Path(vocals_raw)
    if not vocals_path.exists():
        raise HTTPException(410, "Vocals stem file has been cleaned up")

    opts = body or LyricsOptions()
    resolved = _resolve_lyrics_options(
        model=opts.model,
        language=opts.language,
        reference_lyrics=opts.reference_lyrics,
        title=opts.title or (Path(sep_job.filename).stem if sep_job.filename else None),
        use_llm_cleanup=opts.use_llm_cleanup,
    )
    assert resolved.model is not None

    job = LyricsJobRecord(
        filename=vocals_path.name,
        input_path=str(vocals_path),
        owns_input=False,
        model=resolved.model,
        language=resolved.language,
        reference_lyrics=resolved.reference_lyrics,
        title=resolved.title,
        use_llm_cleanup=resolved.use_llm_cleanup,
    )
    job.output_dir = str(Path(settings.lyrics_dir) / job.job_id)

    try:
        await queue.enqueue(job)
    except RuntimeError as exc:
        raise HTTPException(503, "Service is shutting down") from exc

    logger.info("Lyrics from-job accepted  lyrics=%s  sep_job=%s", job.job_id, sep_job_id)
    return _lyrics_to_response(job)


@router.get("/{lyrics_id}", response_model=LyricsJobResponse)
async def get_lyrics_job(lyrics_id: str) -> LyricsJobResponse:
    """Retrieve status, progress, summary, and artifact paths for a lyrics job."""
    queue = _lyrics_queue()
    job = await queue.get_job(lyrics_id)
    if job is None:
        raise HTTPException(404, f"Lyrics job not found: {lyrics_id}")
    return _lyrics_to_response(job)


@router.get("/{lyrics_id}/artifacts/{kind}")
async def download_lyrics_artifact(lyrics_id: str, kind: str) -> FileResponse:
    """Download a generated lyrics artifact (``json``, ``lrc``, ``ass``, ``srt``)."""
    if kind not in ARTIFACT_MEDIA_TYPES:
        raise HTTPException(404, f"Unknown lyrics artifact kind: {kind}")

    queue = _lyrics_queue()
    job = await queue.get_job(lyrics_id)
    if job is None:
        raise HTTPException(404, f"Lyrics job not found: {lyrics_id}")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(409, f"Lyrics job is not completed (status: {job.status.value})")
    if kind not in job.artifact_paths:
        raise HTTPException(404, f"Artifact '{kind}' not found for lyrics job")

    artifact_path = Path(job.artifact_paths[kind])
    if not artifact_path.exists():
        raise HTTPException(410, "Lyrics artifact has been cleaned up")

    media_type, default_name = ARTIFACT_MEDIA_TYPES[kind]
    return FileResponse(
        path=artifact_path,
        media_type=media_type,
        filename=default_name,
    )


@router.delete("/{lyrics_id}", status_code=204)
async def delete_lyrics_job(lyrics_id: str) -> None:
    """Delete a lyrics job and remove its generated artifacts from disk."""
    queue = _lyrics_queue()
    try:
        deleted = await queue.delete_job(lyrics_id)
    except JobProcessingError as exc:
        raise HTTPException(409, str(exc)) from exc
    if not deleted:
        raise HTTPException(404, f"Lyrics job not found: {lyrics_id}")

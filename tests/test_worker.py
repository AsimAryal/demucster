"""Unit tests for the async job queue and worker."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from demucster.models.schemas import JobStatus
from demucster.worker.queue import JobQueue, JobRecord


@pytest.fixture()
def _no_torch_cuda():
    """Ensure torch.cuda.OutOfMemoryError exists even without CUDA."""
    import torch

    if not hasattr(torch.cuda, "OutOfMemoryError"):
        torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
    yield


@pytest.mark.asyncio()
class TestJobQueue:
    """In-memory job queue behaviour."""

    async def test_enqueue_and_retrieve(self):
        q = JobQueue(max_concurrent=1)
        job = JobRecord(model="htdemucs", filename="test.wav")
        await q.enqueue(job)

        retrieved = await q.get_job(job.job_id)
        assert retrieved is not None
        assert retrieved.status == JobStatus.QUEUED
        assert retrieved.filename == "test.wav"

    async def test_list_jobs_filtered_by_status(self):
        q = JobQueue(max_concurrent=1)
        j1 = JobRecord(model="htdemucs")
        j2 = JobRecord(model="htdemucs")
        j2.status = JobStatus.COMPLETED
        await q.enqueue(j1)
        await q.enqueue(j2)

        queued = await q.list_jobs(status=JobStatus.QUEUED)
        assert len(queued) == 1

        completed = await q.list_jobs(status=JobStatus.COMPLETED)
        assert len(completed) == 1

    async def test_delete_removes_job(self):
        q = JobQueue(max_concurrent=1)
        job = JobRecord(model="htdemucs")
        await q.enqueue(job)
        deleted = await q.delete_job(job.job_id)
        assert deleted is True
        assert await q.get_job(job.job_id) is None

    async def test_delete_nonexistent_returns_false(self):
        q = JobQueue(max_concurrent=1)
        assert await q.delete_job("nonexistent") is False

    async def test_progress_update(self):
        q = JobQueue(max_concurrent=1)
        job = JobRecord(model="htdemucs")
        await q.enqueue(job)
        await q.update_progress(job.job_id, 55.5)

        updated = await q.get_job(job.job_id)
        assert updated is not None
        assert updated.progress_pct == 55.5

    async def test_progress_clamped_at_100(self):
        q = JobQueue(max_concurrent=1)
        job = JobRecord(model="htdemucs")
        await q.enqueue(job)
        await q.update_progress(job.job_id, 150.0)

        updated = await q.get_job(job.job_id)
        assert updated is not None
        assert updated.progress_pct == 100.0

    async def test_worker_processes_job(self, tmp_path: Path, _no_torch_cuda: None):
        """Verify the worker picks up a queued job and runs the process fn."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        def process_fn(job: JobRecord, progress_cb):
            # Simulate writing stems
            stems = {}
            for stem in ["vocals", "drums"]:
                p = Path(job.output_dir) / f"{stem}.mp3"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"\x00" * 10)
                stems[stem] = p
            progress_cb(100.0)
            return stems

        q = JobQueue(max_concurrent=2)
        await q.start(process_fn=process_fn)

        job = JobRecord(
            model="htdemucs",
            output_dir=str(output_dir / "job1"),
            input_path=str(tmp_path / "dummy.wav"),
        )
        await q.enqueue(job)

        # Wait for processing
        for _ in range(100):
            j = await q.get_job(job.job_id)
            if j and j.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                break
            await asyncio.sleep(0.05)

        final = await q.get_job(job.job_id)
        assert final is not None
        assert final.status == JobStatus.COMPLETED
        assert "vocals" in final.stem_paths
        assert "drums" in final.stem_paths

        await q.stop()

    async def test_worker_handles_failure(self, _no_torch_cuda: None):
        """If the process fn raises, the job should be marked FAILED."""

        def failing_fn(job: JobRecord, progress_cb):
            raise RuntimeError("Something went wrong")

        q = JobQueue(max_concurrent=1)
        await q.start(process_fn=failing_fn)

        job = JobRecord(model="htdemucs", output_dir="/tmp/out", input_path="/tmp/in.wav")
        await q.enqueue(job)

        for _ in range(100):
            j = await q.get_job(job.job_id)
            if j and j.status == JobStatus.FAILED:
                break
            await asyncio.sleep(0.05)

        final = await q.get_job(job.job_id)
        assert final is not None
        assert final.status == JobStatus.FAILED
        assert "Something went wrong" in (final.error_message or "")

        await q.stop()

    async def test_active_and_queued_counts(self):
        q = JobQueue(max_concurrent=2)
        assert q.active_count == 0
        assert q.queued_count == 0

        for _ in range(3):
            job = JobRecord(model="htdemucs")
            await q.enqueue(job)

        assert q.queued_count == 3

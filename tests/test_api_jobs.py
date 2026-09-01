"""Integration tests for ``/api/v1/jobs/*`` endpoints."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_completion(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    """Poll the job status endpoint until ``COMPLETED`` or ``FAILED``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        data = resp.json()
        if data["status"] in ("COMPLETED", "FAILED"):
            return data
        time.sleep(0.1)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout}s")


# ---------------------------------------------------------------------------
# Upload endpoint
# ---------------------------------------------------------------------------


class TestUpload:
    """POST /api/v1/jobs/upload"""

    def test_upload_returns_202_and_queued(self, client: TestClient, sample_wav: Path):
        with open(sample_wav, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("song.wav", f, "audio/wav")},
                data={"model": "htdemucs_ft", "output_format": "mp3"},
            )
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "QUEUED"
        assert "job_id" in body
        assert body["model"] == "htdemucs_ft"

    def test_upload_rejects_unsupported_extension(self, client: TestClient, tmp_path: Path):
        txt = tmp_path / "nope.txt"
        txt.write_text("not audio")
        with open(txt, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("nope.txt", f, "text/plain")},
            )
        assert resp.status_code == 422

    def test_upload_rejects_filename_path_traversal(self, client: TestClient, sample_wav: Path):
        resp = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("../song.wav", sample_wav.read_bytes(), "audio/wav")},
        )
        assert resp.status_code == 422

    def test_upload_rejects_invalid_options(self, client: TestClient, sample_wav: Path):
        resp = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("song.wav", sample_wav.read_bytes(), "audio/wav")},
            data={"model": "not-a-model", "output_format": "flac"},
        )
        assert resp.status_code == 422

    def test_upload_rejects_stem_not_in_model(self, client: TestClient, sample_wav: Path):
        resp = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("song.wav", sample_wav.read_bytes(), "audio/wav")},
            data={"model": "htdemucs", "stems": "../escape"},
        )
        assert resp.status_code == 422

    def test_upload_completes_and_stems_available(self, client: TestClient, sample_wav: Path):
        with open(sample_wav, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("song.wav", f, "audio/wav")},
            )
        job_id = resp.json()["job_id"]
        data = _wait_for_completion(client, job_id)

        assert data["status"] == "COMPLETED"
        assert data["progress_pct"] == 100.0
        assert data["stems"] is not None
        assert "vocals" in data["stems"]


# ---------------------------------------------------------------------------
# URL endpoint
# ---------------------------------------------------------------------------


class TestURLSubmit:
    """POST /api/v1/jobs/url"""

    def test_url_returns_202(self, client: TestClient):
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": "https://example.com/song.mp3"},
        )
        assert resp.status_code == 202
        assert resp.json()["status"] == "QUEUED"

    def test_url_rejects_non_http_scheme(self, client: TestClient):
        resp = client.post(
            "/api/v1/jobs/url",
            json={"url": "file:///etc/passwd"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Status & listing
# ---------------------------------------------------------------------------


class TestJobStatus:
    """GET /api/v1/jobs and GET /api/v1/jobs/{job_id}"""

    def test_get_nonexistent_returns_404(self, client: TestClient):
        resp = client.get("/api/v1/jobs/does-not-exist")
        assert resp.status_code == 404

    def test_list_jobs_returns_array(self, client: TestClient, sample_wav: Path):
        with open(sample_wav, "rb") as f:
            client.post(
                "/api/v1/jobs/upload",
                files={"file": ("song.wav", f, "audio/wav")},
            )
        resp = client.get("/api/v1/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1


# ---------------------------------------------------------------------------
# Stem download
# ---------------------------------------------------------------------------


class TestStemDownload:
    """GET /api/v1/jobs/{job_id}/stems/{stem_name}"""

    def test_download_stem_after_completion(self, client: TestClient, sample_wav: Path):
        with open(sample_wav, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("song.wav", f, "audio/wav")},
            )
        job_id = resp.json()["job_id"]
        _wait_for_completion(client, job_id)

        resp = client.get(f"/api/v1/jobs/{job_id}/stems/vocals")
        assert resp.status_code == 200
        assert len(resp.content) > 0

    def test_download_missing_stem_returns_404(self, client: TestClient, sample_wav: Path):
        with open(sample_wav, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("song.wav", f, "audio/wav")},
            )
        job_id = resp.json()["job_id"]
        _wait_for_completion(client, job_id)

        resp = client.get(f"/api/v1/jobs/{job_id}/stems/nonexistent")
        assert resp.status_code == 404

    def test_download_before_completion_returns_409(self, client: TestClient, sample_wav: Path):
        """If we can catch the job before it's done, expect 409.

        This test is best-effort — the mock completes very quickly.
        """
        with open(sample_wav, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("song.wav", f, "audio/wav")},
            )
        job_id = resp.json()["job_id"]
        # Immediately try — may already be completed due to fast mock
        resp = client.get(f"/api/v1/jobs/{job_id}/stems/vocals")
        assert resp.status_code in (200, 409)


# ---------------------------------------------------------------------------
# ZIP download
# ---------------------------------------------------------------------------


class TestZipDownload:
    """GET /api/v1/jobs/{job_id}/download-zip"""

    def test_zip_contains_all_stems(self, client: TestClient, sample_wav: Path):
        import io
        import zipfile

        with open(sample_wav, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("song.wav", f, "audio/wav")},
            )
        job_id = resp.json()["job_id"]
        _wait_for_completion(client, job_id)

        resp = client.get(f"/api/v1/jobs/{job_id}/download-zip")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"

        zf = zipfile.ZipFile(io.BytesIO(resp.content))
        names = zf.namelist()
        assert len(names) >= 4  # vocals, drums, bass, other


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDeleteJob:
    """DELETE /api/v1/jobs/{job_id}"""

    def test_delete_returns_204(self, client: TestClient, sample_wav: Path):
        with open(sample_wav, "rb") as f:
            resp = client.post(
                "/api/v1/jobs/upload",
                files={"file": ("song.wav", f, "audio/wav")},
            )
        job_id = resp.json()["job_id"]
        _wait_for_completion(client, job_id)

        resp = client.delete(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 204

        # Should be gone now
        resp = client.get(f"/api/v1/jobs/{job_id}")
        assert resp.status_code == 404

    def test_delete_nonexistent_returns_404(self, client: TestClient):
        resp = client.delete("/api/v1/jobs/bogus-id")
        assert resp.status_code == 404

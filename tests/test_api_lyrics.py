"""Integration tests for ``/api/v1/lyrics/*`` endpoints."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi.testclient import TestClient


def _wait_for_lyrics(client: TestClient, lyrics_id: str, timeout: float = 10.0) -> dict:
    """Poll lyrics status until ``COMPLETED`` or ``FAILED``."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/v1/lyrics/{lyrics_id}")
        data = resp.json()
        if data["status"] in ("COMPLETED", "FAILED"):
            return data
        time.sleep(0.05)
    raise TimeoutError(f"Lyrics job {lyrics_id} did not finish within {timeout}s")


def _wait_for_sep(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = client.get(f"/api/v1/jobs/{job_id}")
        data = resp.json()
        if data["status"] in ("COMPLETED", "FAILED"):
            return data
        time.sleep(0.05)
    raise TimeoutError(f"Separation job {job_id} did not finish within {timeout}s")


def test_health_reports_lyrics_capabilities(client: TestClient) -> None:
    body = client.get("/api/v1/health").json()
    assert body["lyrics_enabled"] is True
    assert body["lyrics_active_jobs"] >= 0
    assert body["lyrics_queued_jobs"] >= 0


def test_upload_lyrics_completes_and_serves_artifacts(client: TestClient, sample_wav: Path) -> None:
    with open(sample_wav, "rb") as f:
        resp = client.post(
            "/api/v1/lyrics/upload",
            files={"file": ("vocals.wav", f, "audio/wav")},
            data={
                "model": "large-v3-turbo",
                "language": "en",
                "title": "Demo Track",
                "reference_lyrics": "Hello karaoke world",
                "use_llm_cleanup": "true",
            },
        )
    assert resp.status_code == 202
    lyrics_id = resp.json()["job_id"]
    done = _wait_for_lyrics(client, lyrics_id)

    assert done["status"] == "COMPLETED"
    assert done["summary"]["word_count"] == 3
    assert done["summary"]["aligned_with_reference"] is True
    assert set(done["artifacts"]) == {"json", "lrc", "ass", "srt"}

    for kind in ("json", "lrc", "ass", "srt"):
        dl = client.get(f"/api/v1/lyrics/{lyrics_id}/artifacts/{kind}")
        assert dl.status_code == 200
        assert len(dl.content) > 0

    deleted = client.delete(f"/api/v1/lyrics/{lyrics_id}")
    assert deleted.status_code == 204
    assert client.get(f"/api/v1/lyrics/{lyrics_id}").status_code == 404


def test_lyrics_upload_validation(client: TestClient, sample_wav: Path) -> None:
    bad = client.post(
        "/api/v1/lyrics/upload",
        files={"file": ("../vocals.wav", sample_wav.read_bytes(), "audio/wav")},
    )
    assert bad.status_code == 422

    bad_model = client.post(
        "/api/v1/lyrics/upload",
        files={"file": ("vocals.wav", sample_wav.read_bytes(), "audio/wav")},
        data={"model": "unknown-whisper"},
    )
    assert bad_model.status_code == 422


def test_lyrics_from_separation_job(client: TestClient, sample_wav: Path) -> None:
    with open(sample_wav, "rb") as f:
        sep = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("song.wav", f, "audio/wav")},
            data={"stems": "vocals,drums"},
        )
    sep_id = sep.json()["job_id"]
    _wait_for_sep(client, sep_id)

    res = client.post(
        f"/api/v1/lyrics/from-job/{sep_id}",
        json={"model": "medium", "use_llm_cleanup": False},
    )
    assert res.status_code == 202
    done = _wait_for_lyrics(client, res.json()["job_id"])
    assert done["status"] == "COMPLETED"
    assert done["model"] == "medium"

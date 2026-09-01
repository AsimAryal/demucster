"""Tests for ``/api/v1/health`` and ``/api/v1/models``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi.testclient import TestClient


class TestHealth:
    """GET /api/v1/health"""

    def test_returns_200_with_expected_fields(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        body = resp.json()

        assert body["status"] == "healthy"
        assert "cuda_available" in body
        assert "active_jobs" in body
        assert "queued_jobs" in body

    def test_job_counts_are_non_negative(self, client: TestClient):
        body = client.get("/api/v1/health").json()
        assert body["active_jobs"] >= 0
        assert body["queued_jobs"] >= 0


class TestModels:
    """GET /api/v1/models"""

    def test_returns_list_of_models(self, client: TestClient):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)
        assert len(models) >= 6  # htdemucs_ft, htdemucs, htdemucs_6s, mdx_extra, mdx_extra_q, mdx

    def test_default_model_is_flagged(self, client: TestClient):
        models = client.get("/api/v1/models").json()
        defaults = [m for m in models if m["is_default"]]
        assert len(defaults) == 1
        assert defaults[0]["name"] == "htdemucs_ft"

    def test_each_model_has_stems(self, client: TestClient):
        models = client.get("/api/v1/models").json()
        for m in models:
            assert "stems" in m
            assert len(m["stems"]) >= 4

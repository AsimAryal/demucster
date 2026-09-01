"""Top-level API router aggregation."""

from __future__ import annotations

from fastapi import APIRouter

from demucster.api.health import router as health_router
from demucster.api.jobs import router as jobs_router

api_router = APIRouter()
api_router.include_router(jobs_router)
api_router.include_router(health_router)

# CLAUDE.md — Claude Code Context

> This file provides context for Claude Code when working on Demucster.

## Project Overview

Demucster is a GPU-accelerated audio stem separation microservice built with:
- **FastAPI** (async, Pydantic v2)
- **PyTorch ≥ 2.7** (CUDA 12.8, Blackwell sm_12x)
- **Demucs** (htdemucs_ft, htdemucs, htdemucs_6s, mdx_extra, mdx_extra_q, mdx)
- **uv** (package manager)
- **ruff** (linter + formatter)

Target hardware: **NVIDIA DGX Spark** (GB10 Blackwell, 128 GB unified LPDDR5x).

## Architecture Quick Reference

```
src/demucster/
├── main.py          ← App factory, lifespan (startup/shutdown), process_fn
├── api/             ← FastAPI routes (jobs.py, health.py)
├── core/            ← Config (Pydantic Settings), GPU utils, logging
├── models/          ← Pydantic v2 schemas (NO dataclasses)
├── services/        ← Demucs wrapper, audio utils, yt-dlp, cleanup
└── worker/          ← Async job queue with bounded GPU semaphore
```

## Critical Conventions

1. **Pydantic everywhere** — All models use `pydantic.BaseModel`, never `dataclass`.
2. **Type annotations** — All functions have full type hints.
3. **Ruff** — Lint with `ruff check`, format with `ruff format`. Config in `pyproject.toml`.
4. **uv** — Use `uv sync`, `uv run`, `uv pip install`. Never raw `pip`.
5. **src/ layout** — Package at `src/demucster/`, import as `demucster.*`.
6. **Lazy imports** — Heavy deps (`demucs`, `yt_dlp`) are imported inside functions.

## Key Design Decisions

### GPU Memory (GB10 Blackwell)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — set in gpu.py and Dockerfile
- Do NOT set `CUDA_MANAGED_FORCE_DEVICE_ALLOC` — coherent driver handles it
- Health endpoint uses `torch.cuda` APIs, NOT `pynvml` (NVML misreports on GB10)

### Job Queue
- In-process `asyncio.Queue` + `asyncio.Semaphore` (no Redis/Celery)
- Inference runs in `loop.run_in_executor(None, ...)` — keeps event loop responsive
- Progress callbacks bridge thread→async via `asyncio.run_coroutine_threadsafe`
- CUDA OOM caught and handled gracefully (job fails, queue continues)

### Demucs Integration
- New `demucs.api.Separator` per job (thread-safe, avoids callback conflicts)
- Model weights cached by demucs internally — no double-loading
- Stems saved via `torchaudio.save` (WAV) or `ffmpeg` subprocess (MP3)

## Common Commands

```bash
# Dev server
uv run uvicorn demucster.main:app --reload --port 8000

# Tests (mocked — no GPU needed)
uv run pytest tests/ -v

# Lint + format
uv run ruff check --fix src/ tests/
uv run ruff format src/ tests/

# Docker
docker compose up --build
```

## Test Strategy

- All tests mock `DemucsService.separate`, `torch.cuda`, and `torchaudio.info`
- `TestClient` (sync) from FastAPI handles lifespan
- `conftest.py` patches env vars, clears `get_settings` cache
- Worker tests use `pytest-asyncio` with real `asyncio.Queue`

## File Quick Reference

| File | Purpose |
|---|---|
| `core/config.py` | `Settings` (Pydantic BaseSettings), `get_settings()` singleton |
| `core/gpu.py` | `get_device()`, `get_gpu_info()`, allocator env setup |
| `models/schemas.py` | `JobStatus`, `SeparationOptions`, `JobResponse`, `HealthResponse` |
| `worker/queue.py` | `JobRecord`, `JobQueue` (async queue + semaphore) |
| `services/separator.py` | `DemucsService`, `SUPPORTED_MODELS` dict |
| `services/audio.py` | `validate_audio_file()`, `save_stem()` |
| `services/downloader.py` | `download_audio()` via yt-dlp |
| `services/cleanup.py` | `run_cleanup_loop()` background task |
| `api/jobs.py` | Upload, URL, status, stem download, ZIP, delete endpoints |
| `api/health.py` | Health check + model listing |
| `main.py` | `create_app()`, `lifespan()`, `_process_job()` |

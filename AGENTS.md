# AGENTS.md — AI Agent Rules for Demucster

> Rules and context for all AI coding agents working on this project.

## Project Identity

**Demucster** — A GPU-accelerated audio stem separation microservice.

- **Language:** Python 3.11+
- **Framework:** FastAPI (async)
- **ML Runtime:** PyTorch ≥ 2.7 + CUDA ≥ 12.8
- **Target Hardware:** NVIDIA DGX Spark (GB10 Blackwell, 128 GB unified memory)
- **Package Manager:** `uv` (never use raw `pip`)
- **Linter/Formatter:** `ruff` (config in `pyproject.toml`)
- **Build Backend:** `hatchling`
- **Test Framework:** `pytest` + `pytest-asyncio`

---

## Mandatory Rules

### Code Style
- **Pydantic v2 `BaseModel`** for all data models. Never use `dataclasses`.
- **Full type annotations** on all function signatures and return types.
- **Docstrings** on all public classes and functions (Google or NumPy style).
- **Line length:** 100 characters max.
- Run `ruff check` and `ruff format` before committing.

### Package Layout
- Source code lives in `src/demucster/` (src layout).
- Import paths are `demucster.*` (e.g., `from demucster.core.config import get_settings`).
- Tests live in `tests/` at the project root.

### Dependencies
- Use `uv sync` to install, `uv run` to execute.
- Local PyTorch is resolved from the CUDA 12.8 index; Docker preserves NGC's PyTorch.
- Heavy dependencies (`demucs`, `yt_dlp`) are **lazy-imported** inside functions to avoid import-time crashes when the library isn't installed.

### GPU / Blackwell
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set in `core/gpu.py` and `Dockerfile`.
- Do **NOT** set `CUDA_MANAGED_FORCE_DEVICE_ALLOC` — the Grace Blackwell coherent driver handles memory residency.
- Use `torch.cuda` APIs for GPU introspection, **NOT** `pynvml` or `nvidia-smi` (NVML misreports on GB10 unified memory).

### Error Handling
- CUDA OOM: catch `torch.cuda.OutOfMemoryError`, call `torch.cuda.empty_cache()`, set job status to `FAILED`, continue processing queue.
- Invalid audio: raise `InvalidAudioError` (custom exception in `services/audio.py`).
- Download failures: raise `DownloadError` (custom exception in `services/downloader.py`).
- Never crash the worker loop — catch all exceptions per-job.

---

## Architecture Reference

```
src/demucster/
├── main.py              # App factory + lifespan (startup/shutdown)
├── api/
│   ├── router.py        # Route aggregation
│   ├── jobs.py          # POST /upload, POST /url, GET /{id}, DELETE, etc.
│   └── health.py        # GET /health, GET /models
├── core/
│   ├── config.py        # Pydantic Settings (env-driven, lru_cache singleton)
│   ├── gpu.py           # CUDA init, memory info, allocator config
│   └── logging.py       # Structured JSON logging
├── models/
│   └── schemas.py       # Request/response Pydantic models + enums
├── services/
│   ├── separator.py     # Demucs inference wrapper + SUPPORTED_MODELS
│   ├── audio.py         # Validation, save_stem (WAV/MP3 via ffmpeg)
│   ├── downloader.py    # yt-dlp URL fetcher
│   └── cleanup.py       # Background file retention purge
└── worker/
    └── queue.py         # Async job queue (asyncio.Queue + Semaphore)
```

### Key Patterns

1. **Job Queue** — `asyncio.Queue` + `asyncio.Semaphore(MAX_CONCURRENT_JOBS)`. Worker loop pulls jobs, spawns bounded tasks, runs inference in `run_in_executor`.
2. **Progress Bridge** — Demucs callback runs in executor thread, schedules async progress updates via `asyncio.run_coroutine_threadsafe`.
3. **Settings** — `get_settings()` is an `lru_cache(1)` function returning `pydantic_settings.BaseSettings`. Clear cache in tests.
4. **Module globals** — `main.py` exposes `job_queue` and `demucs_service` at module level for endpoint access (set during lifespan).

---

## Commands

```bash
# Install
uv sync

# Dev server
uv run uvicorn demucster.main:app --reload --port 8000

# Tests (all mocked, no GPU needed)
uv run pytest tests/ -v

# Coverage
uv run pytest tests/ --cov=demucster --cov-report=term-missing

# Lint
uv run ruff check src/ tests/
uv run ruff check --fix src/ tests/

# Format
uv run ruff format src/ tests/

# Docker
docker compose up --build
```

---

## Testing Rules

- **Never require a GPU** — tests mock `DemucsService.separate` and `torch.cuda.*`.
- **TestClient** from FastAPI handles lifespan (startup/shutdown).
- **`conftest.py`** patches env vars and clears `get_settings.cache_clear()` to isolate tests.
- **Worker tests** use real `asyncio.Queue` with fake `process_fn`.
- **Service tests** mock `yt_dlp` and filesystem operations.

---

## Supported Demucs Models

| Model | Stems | Notes |
|---|---|---|
| `htdemucs_ft` ⭐ | vocals, drums, bass, other | Highest quality, default |
| `htdemucs` | vocals, drums, bass, other | Fast, strong quality |
| `htdemucs_6s` | vocals, drums, bass, other, guitar, piano | 6-stem |
| `mdx_extra` | vocals, drums, bass, other | Alternative architecture |
| `mdx_extra_q` | vocals, drums, bass, other | Quantised, lightest |
| `mdx` | vocals, drums, bass, other | MDX baseline |

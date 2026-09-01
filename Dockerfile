# =============================================================================
# Demucster — GPU-accelerated audio stem separation
# =============================================================================
# Base: NVIDIA NGC PyTorch with CUDA 13 and pre-compiled Blackwell kernels
# for the DGX Spark GB10 (Blackwell) architecture.
#
# Key decisions:
#   • Single-stage using NGC image (PyTorch + CUDA already present).
#   • uv installed for fast dependency resolution.
#   • ffmpeg + libsndfile for audio format support.
#   • Non-root runtime user.
#   • PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to prevent
#     fragmentation OOM on the 128 GB unified memory pool.
#   • NO CUDA_MANAGED_FORCE_DEVICE_ALLOC — the Grace Blackwell coherent
#     driver manages residency natively.
# =============================================================================

FROM nvcr.io/nvidia/pytorch:26.07-py3

LABEL maintainer="Demucster Team"
LABEL description="GPU-accelerated Demucs stem separation microservice"

# --- System dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cargo \
        cmake \
        ffmpeg \
        libsndfile1 \
        libopus-dev \
        pkg-config \
        rustc \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv package manager ---------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.8.3 /uv /uvx /usr/local/bin/

# --- Application code -------------------------------------------------------
WORKDIR /app

# Copy project metadata first for layer caching. README is required by Hatchling.
COPY pyproject.toml requirements-container.txt README.md .python-version ./

# Copy source tree
COPY src/ src/

# Isolate application packages while inheriting NGC's system torch/torchaudio.
# The no-deps export cannot resolve or replace the NVIDIA framework packages.
RUN python -m venv --system-site-packages /opt/demucster \
    && base_torch="$(/opt/demucster/bin/python -c 'import torch; print(torch.__version__)')" \
    && base_torchaudio="$(/opt/demucster/bin/python -c 'import torchaudio; print(torchaudio.__version__)')" \
    && uv pip install --python /opt/demucster/bin/python --no-cache --no-deps --require-hashes \
        -r requirements-container.txt \
    && uv pip install --python /opt/demucster/bin/python --no-cache --no-deps -e . \
    && test "$(/opt/demucster/bin/python -c 'import torch; print(torch.__version__)')" = "${base_torch}" \
    && test "$(/opt/demucster/bin/python -c 'import torchaudio; print(torchaudio.__version__)')" = "${base_torchaudio}"

# --- Environment variables --------------------------------------------------
# GB10 Blackwell allocator tuning
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NVIDIA container runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Demucster defaults (overridable via docker-compose / -e flags)
ENV DEFAULT_MODEL=htdemucs_ft
ENV DEVICE=cuda
ENV UPLOAD_DIR=/data/uploads
ENV OUTPUT_DIR=/data/outputs
ENV MAX_CONCURRENT_JOBS=2
ENV RETENTION_HOURS=24
ENV LOG_LEVEL=INFO
ENV PREWARM_MODEL=true
ENV REQUIRE_CUDA=true

# --- Non-root user -----------------------------------------------------------
RUN useradd --create-home --shell /bin/bash demucster \
    && mkdir -p /data/uploads /data/outputs \
    && chown -R demucster:demucster /data /app

USER demucster

# --- Entrypoint --------------------------------------------------------------
EXPOSE 8000

CMD ["/opt/demucster/bin/python", "-m", "uvicorn", "demucster.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]

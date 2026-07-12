# ============================================================
# Falcon-7B Medical QA — Production Inference Server
# ============================================================
# Base: PyTorch + CUDA 12.1 + cuDNN 8 (matches Colab T4)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Metadata labels
LABEL maintainer="singularity-14"
LABEL description="Falcon-7B-Instruct fine-tuned on MedQA-USMLE — FastAPI inference server"
LABEL version="1.0.0"

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── Set working directory ──────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy only requirements first for Docker layer caching
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# ── Copy source code ───────────────────────────────────────────────────────────
COPY src/ ./src/
COPY configs/ ./configs/

# ── Model artifacts (volume-mounted at runtime — not baked into image) ─────────
# Place your peft_adapter/ folder at: docker/model/peft_adapter/
# Then mount with: -v $(pwd)/model:/app/model
RUN mkdir -p /app/model/peft_adapter

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_HOME=/app/.cache/huggingface

# ── Cache directory ────────────────────────────────────────────────────────────
RUN mkdir -p /app/.cache/huggingface

# ── Expose API port ────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
CMD ["python", "-m", "uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

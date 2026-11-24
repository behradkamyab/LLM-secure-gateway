FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    GATEWAY_CONFIG=configs/e0_zeroshot.yaml

# OpenMP runtime needed by faiss-cpu on slim Linux images
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the official uv binary
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy package metadata, README (needed by hatchling), and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install production inference dependencies into system Python
RUN uv pip install --system --no-cache ".[infer]"

# Copy runtime YAML config and router FAQ intents
COPY configs/ ./configs/
COPY data/router_intents.json ./data/router_intents.json

# Run as a non-root user with a writable Hugging Face cache directory
RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5)" || exit 1

CMD ["uvicorn", "gateway.infer.main:app", "--host", "0.0.0.0", "--port", "8000"]

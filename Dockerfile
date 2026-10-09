FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# The /ask endpoint needs retrieval (pgvector client, BM25, sentence-transformers). CPU-only
# PyTorch keeps the image small; models are mounted at /app/models (see docker-compose.yml).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install ".[retrieval]"

RUN useradd --create-home app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "papertrail.api:app", "--host", "0.0.0.0", "--port", "8000"]

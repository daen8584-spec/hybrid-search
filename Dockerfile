# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src src
# The binary psycopg build bundles libpq, so the image needs no system packages.
RUN pip install . "psycopg[binary]>=3.2"

RUN useradd --system --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"]
# Migrations are idempotent and serialized by an advisory lock, so every start applies
# any pending ones before serving.
CMD ["sh", "-c", "hybrid-search migrate && exec hybrid-search serve --host 0.0.0.0 --port 8000"]

FROM python:3.11-slim as builder

# Installer uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copier UNIQUEMENT les requirements (layer cache optimal)
COPY requirements/base.txt requirements/prod.txt requirements/monitoring.txt ./

# Créer venv et installer dépendances
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN uv pip install \
    -r base.txt \
    -r prod.txt \
    -r monitoring.txt

FROM python:3.11-slim

# Installer curl pour healthcheck
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier le venv pré-installé depuis le builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier le code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY config/ ./config/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "scripts/run_api.py"]
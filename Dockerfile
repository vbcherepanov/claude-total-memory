# syntax=docker/dockerfile:1.7
# Multi-stage build for total-agent-memory
# Single image runs MCP HTTP + dashboard + reflection via a small supervisor
# (docker/tam_supervisor.py). Compose still works — each service overrides
# command/TAM_SUPERVISOR_SERVICES to run one process at a time.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

# ─────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TAM_MEMORY_DIR=/data \
    DASHBOARD_PORT=37737 \
    DASHBOARD_BIND=0.0.0.0 \
    TAM_SUPERVISOR_SERVICES=mcp,dashboard,reflection,scheduler \
    MCP_TRANSPORT=http \
    MCP_HTTP_HOST=0.0.0.0 \
    MCP_HTTP_PORT=3737 \
    # Pin the model cache to the data volume — /tmp is often a tmpfs (~64MB)
    # in container runtimes, and a purged fastembed cache is the usual reason
    # a memory server suddenly re-downloads its model on every start.
    # fastembed reads neither HF_HOME nor XDG_CACHE_HOME: its cache is
    # FASTEMBED_CACHE_PATH, which paths.pin_model_cache() derives from
    # TAM_MODEL_CACHE. Without this the ONNX model lands in the container's
    # temp dir and is re-fetched on every `docker run` — the whole point of
    # the volume. HF_HOME/XDG_CACHE_HOME still steer the hub's own scratch
    # files (xet chunks, onnxruntime), so they stay.
    # The torch-specific caches went with the torch stack: the image installs
    # requirements.txt, which no longer resolves torch (see the [rerank] extra).
    TAM_MODEL_CACHE=/data/models \
    HF_HOME=/data/.cache/huggingface \
    XDG_CACHE_HOME=/data/.cache \
    # Try host Ollama first (works on Docker Desktop / Windows / WSL2).
    # Server falls back to FTS+embeddings if Ollama unreachable —
    # nothing crashes, LLM stages just skip.
    OLLAMA_URL=http://host.docker.internal:11434 \
    USE_OLLAMA_EMBED=auto \
    MEMORY_LLM_ENABLED=auto

# curl for healthcheck, tini for clean signal handling
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Non-root user (kept as `memory` — `tam` was a leftover from an
# earlier build; keeping `memory` matches repo Dockerfile + compose).
RUN useradd -m -u 1000 memory && \
    mkdir -p /data && chown memory:memory /data

COPY --from=builder /install /usr/local
COPY --chown=memory:memory src/ ./src/
COPY --chown=memory:memory migrations/ ./migrations/
COPY --chown=memory:memory docker/ ./docker/
# Runtime data the server resolves relative to the repo root. Missing here
# meant canonical tag normalisation and save-filters were dead in the image.
COPY --chown=memory:memory vocabularies/ ./vocabularies/
COPY --chown=memory:memory filters/ ./filters/

# Convenience entrypoint shim — `docker run … mcp|dashboard|reflection|all`
# routes through the supervisor (which respects TAM_SUPERVISOR_SERVICES).
# `docker run -i … stdio` bypasses the supervisor and speaks MCP over
# stdin/stdout, which is how Docker MCP Toolkit and any `docker run -i`
# client expect a local server to behave. Logs go to stderr, so they cannot
# corrupt the JSON-RPC stream on stdout.
RUN printf '#!/bin/sh\nset -e\ncase "$1" in\n  stdio)\n    MCP_TRANSPORT=stdio exec python -m src.server\n    ;;\n  mcp|dashboard|reflection|scheduler)\n    TAM_SUPERVISOR_SERVICES="$1" exec python /app/docker/tam_supervisor.py\n    ;;\n  all|"")\n    exec python /app/docker/tam_supervisor.py\n    ;;\n  *)\n    exec "$@"\n    ;;\nesac\n' > /usr/local/bin/tam-entrypoint \
    && chmod +x /usr/local/bin/tam-entrypoint

USER memory

VOLUME ["/data"]
# 3737  = MCP Streamable HTTP (POST/GET/DELETE /mcp + /healthz)
# 37737 = web dashboard (/api/stats + /healthz)
EXPOSE 3737 37737

# Healthcheck: both /healthz endpoints must answer. They're DB-independent
# (don't depend on migrations finishing) so the container is reported
# healthy as soon as both servers are listening — typically <15s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://127.0.0.1:37737/healthz >/dev/null \
        && curl -fsS http://127.0.0.1:3737/healthz >/dev/null \
        || exit 1

# tini PID 1 → supervisor → mcp + dashboard + reflection + scheduler
ENTRYPOINT ["/usr/bin/tini", "--", "tam-entrypoint"]
CMD ["all"]

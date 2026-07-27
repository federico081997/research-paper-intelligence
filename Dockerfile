# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS runtime

# FAISS, NumPy, scikit-learn and similar numerical libraries may require
# the OpenMP runtime.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pin uv rather than using the floating "latest" tag.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME="/app/.cache/huggingface"

WORKDIR /app

# Install dependencies in a separate cached layer.
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-project

# Copy the application only after dependencies have been installed.
COPY . .

# Install the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev

# Do not run the application as root.
RUN chmod +x /app/src/research_paper_intelligence/cli/prepare_runtime_data.sh \
    && groupadd --system app \
    && useradd \
        --system \
        --gid app \
        --home-dir /app \
        app \
    && mkdir -p \
        /app/data/raw \
        /app/data/processed \
        /app/data/artifacts \
        /app/.cache/huggingface \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["uvicorn", \
    "research_paper_intelligence.api.app:fastapi_app", \
    "--host", \
    "0.0.0.0", \
    "--port", \
    "8000"]
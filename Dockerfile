# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS runtime

# Runtime dependency required by FAISS, NumPy and scikit-learn.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install a pinned uv version.
COPY --from=ghcr.io/astral-sh/uv:0.11.32 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}" \
    HF_HOME="/app/.cache/huggingface"

WORKDIR /app

# Install third-party dependencies in a cacheable layer.
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev \
        --no-install-project

# Copy the application source.
COPY . .

# Install the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
        --locked \
        --no-dev

# Create a non-root application user.
#
# Only writable runtime directories need to belong to the application user.
# The virtual environment and application source can remain root-owned because
# the application only needs read and execute permissions for them.
RUN groupadd --system app \
    && useradd \
        --system \
        --gid app \
        --home-dir /app \
        --no-create-home \
        app \
    && chmod +x \
        /app/src/research_paper_intelligence/cli/prepare_runtime_data.sh \
    && mkdir -p \
        /app/data/raw \
        /app/data/processed \
        /app/data/artifacts \
        /app/.cache/huggingface \
        /app/.streamlit \
    && chown -R app:app \
        /app/data \
        /app/.cache \
        /app/.streamlit

USER app

EXPOSE 8000

CMD ["uvicorn", \
    "research_paper_intelligence.api.app:fastapi_app", \
    "--host", \
    "0.0.0.0", \
    "--port", \
    "8000"]
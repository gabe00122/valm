FROM python:3.14-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt update && apt install -y build-essential curl git

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY README.md pyproject.toml uv.lock ./
COPY src src

# Sync the project
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

COPY env_assets env_assets
COPY configs configs
COPY start.sh /start.sh

RUN chmod +x /start.sh

ENV PATH="/app/.venv/bin:$PATH"
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

CMD ["/start.sh"]

FROM python:3.13-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# rust deps
RUN apt update && apt install -y build-essential curl git
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY README.md pyproject.toml uv.lock Cargo.toml Cargo.lock ./
COPY src src
COPY python python

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

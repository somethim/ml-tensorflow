FROM python:3.11-slim AS builder-base

WORKDIR /app
COPY pyproject.toml poetry.lock ./

RUN pip install --upgrade pip && \
    pip install poetry --no-cache-dir && \
    poetry config virtualenvs.in-project true

FROM builder-base AS builder-dev
COPY . .
RUN poetry install && \
    rm -rf ~/.cache/pip/* && \
    find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

FROM python:3.11-slim
WORKDIR /app

# Install nodejs for pyright
RUN apt-get update && \
    apt-get install -y nodejs npm && \
    npm install -g pyright && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder-dev /app/.venv /app/.venv
COPY --from=builder-dev /app/poetry.lock /app/pyproject.toml ./
COPY . .

RUN pip install poetry --no-cache-dir && \
    poetry config virtualenvs.in-project true && \
    ln -s /app/.venv/bin/lint /usr/local/bin/lint && \
    ln -s /app/.venv/bin/ml /usr/local/bin/ml && \
    mkdir -p data/{raw,processed} \
    models/{checkpoints,saved_models} \
    logs/tensorboard && \
    rm -rf /root/.cache/pip/* && \
    apt-get clean

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

CMD ["/bin/bash"]
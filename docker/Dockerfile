FROM python:3.11-slim AS builder-base

WORKDIR /app
COPY pyproject.toml ./

RUN pip install --upgrade pip && \
    pip install poetry --no-cache-dir && \
    poetry config virtualenvs.in-project true

FROM builder-base AS builder-dev
COPY . .
RUN poetry install --no-interaction --no-ansi && \
    rm -rf ~/.cache/pip/* && \
    find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

FROM python:3.11-slim
WORKDIR /app

# Copy the entire virtual environment and project files
COPY --from=builder-dev /app/.venv /app/.venv
COPY --from=builder-dev /app/pyproject.toml ./
COPY . .

# Install poetry in the final image and create necessary directories
RUN pip install poetry --no-cache-dir && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi && \
    ln -s /app/.venv/bin/lint /usr/local/bin/lint && \
    ln -s /app/.venv/bin/ml /usr/local/bin/ml && \
    mkdir -p data/{raw,processed} \
    models/{checkpoints,saved_models} \
    logs/tensorboard && \
    rm -rf /root/.cache/pip/*

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

CMD ["/bin/bash"]
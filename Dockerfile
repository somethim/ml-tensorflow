FROM python:3.11-slim-bullseye AS builder-base

WORKDIR /app
COPY pyproject.toml ./

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc6-dev python3-dev && \
    pip install --upgrade pip && \
    pip install poetry --no-cache-dir && \
    poetry config virtualenvs.in-project true

# Development stage with all dependencies
FROM builder-base AS dev
RUN poetry install --no-interaction --no-ansi && \
    # Clean up build dependencies but keep poetry
    apt-get purge -y gcc python3-dev libc6-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf ~/.cache/pip/* /root/.cache/pip/* && \
    # Remove Python cache and unnecessary files
    find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete

WORKDIR /app
COPY . .

# Install poetry and register scripts
RUN pip install poetry --no-cache-dir && \
    poetry config virtualenvs.in-project true && \
    poetry install --no-interaction --no-ansi

# Create necessary directories
RUN mkdir -p data/{raw,processed} \
    models/{checkpoints,saved_models} \
    logs/tensorboard && \
    # Clean up any Python cache from the copied files
    find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete && \
    # Clean up apt in final image
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

# Add poetry and local bin to PATH
RUN echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc && \
    echo 'export PATH="/app/.venv/bin:$PATH"' >> /root/.bashrc

ENTRYPOINT ["/bin/bash"]

# Production stage with only prod dependencies
FROM builder-base AS prod
RUN poetry install --no-interaction --no-ansi --no-dev && \
    # Clean up everything including poetry
    pip uninstall -y poetry && \
    apt-get purge -y gcc python3-dev libc6-dev && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf ~/.cache/pip/* /root/.cache/pip/* && \
    # Remove Python cache and unnecessary files
    find /app/.venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type f -name "*.pyc" -delete && \
    # Remove tests and documentation
    find /app/.venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /app/.venv -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true

WORKDIR /app
COPY . .

# Create necessary directories
RUN mkdir -p data/{raw,processed} \
    models/{checkpoints,saved_models} \
    logs/tensorboard && \
    # Clean up any Python cache from the copied files
    find /app -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /app -type f -name "*.pyc" -delete && \
    # Clean up apt in final image
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"

ENTRYPOINT ["/bin/bash"]
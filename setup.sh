#!/bin/bash

# Exit on error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Setting up ML project...${NC}"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.11.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo -e "${RED}Error: Python 3.11+ is required${NC}"
    exit 1
fi

# Install poetry if not installed
if ! command -v poetry &> /dev/null; then
    echo -e "${BLUE}Installing Poetry...${NC}"
    curl -sSL https://install.python-poetry.org | python3 -
fi

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
poetry install

# Create necessary directories
echo -e "${BLUE}Creating necessary directories...${NC}"
mkdir -p data/{raw,processed}
mkdir -p models/{checkpoints,saved_models}
mkdir -p logs/tensorboard

# Setup pre-commit hooks
echo -e "${BLUE}Setting up pre-commit hooks...${NC}"
poetry run pre-commit install

echo -e "${GREEN}Setup complete! 🚀${NC}"
echo -e "To get started:"
echo -e "1. Activate the virtual environment: ${BLUE}poetry shell${NC}"

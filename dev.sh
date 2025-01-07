#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DOCKER_COMPOSE_FILE="${SCRIPT_DIR}/docker/docker-compose.yml"

# Change to the project root directory
cd "${SCRIPT_DIR}" || {
    echo -e "${RED}Failed to change to project directory: ${SCRIPT_DIR}${NC}"
    exit 1
}

case "$1" in
  "help"|"")
    echo -e "\n${BLUE}ML Development Environment Commands:${NC}"
    echo -e "\nUsage: $0 {help|shell|down}"
    echo -e "\nAvailable commands:"
    echo -e "  ${GREEN}help${NC}   Show this help message"
    echo -e "  ${GREEN}shell${NC}  Open an interactive shell in the container (default)"
    echo -e "  ${GREEN}down${NC}   Stop development environment"
    echo
    ;;
  "shell")
    echo -e "${BLUE}Opening interactive shell in container...${NC}"
    docker compose -f "${DOCKER_COMPOSE_FILE}" run --rm ml-dev bash
    ;;
  "down")
    echo -e "${BLUE}Stopping development environment...${NC}"
    docker compose -f "${DOCKER_COMPOSE_FILE}" down
    ;;
  *)
    echo -e "Usage: $0 {help|shell|down}"
    echo -e "\nCommands:"
    echo -e "  ${GREEN}help${NC}  Show this help message"
    echo -e "  ${GREEN}shell${NC} Open an interactive shell in the container (default)"
    echo -e "  ${GREEN}down${NC}  Stop development environment"
    exit 1
    ;;
esac 
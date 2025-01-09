"""Linting checks module."""

import logging
import subprocess
from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class LintCommand:
    """Lint command configuration.

    Attributes:
        description: Human-readable description of the lint command
        command: List of command arguments to execute
    """

    description: str
    command: List[str] = field(default_factory=list)

    def __init__(self, description: str, command: List[str]) -> None:
        """Initialize a LintCommand.

        Args:
            description: Human-readable description of the lint command
            command: List of command arguments to execute
        """
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "command", command)


def run_command(command: List[str], description: str) -> bool:
    """Run a shell command and print its output."""
    logging.info(f"Running {description}...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        logging.warning(f"{description} failed:")
        logging.warning(result.stdout)
        logging.warning(result.stderr)
        return False
    logging.info(f"{description} passed!")
    return True


def get_linters() -> List[LintCommand]:
    """Get all linter commands with fixing enabled where possible."""
    return [
        LintCommand(
            "Black formatter",
            ["black", "."],
        ),
        LintCommand(
            "isort",
            ["isort", "."],
        ),
        LintCommand(
            "mypy type checking",
            ["mypy", "--config-file", "tools/configs/mypy.ini", "."],
        ),
        LintCommand(
            "flake8 linting",
            ["flake8", "--config", "tools/configs/.flake8"],
        ),
    ]


def run_lint() -> bool:
    """Run all linters with automatic fixing where possible.

    Returns:
        bool: True if all linters passed, False if any failed
    """
    logging.info("Running linters...\n")

    linters = get_linters()
    failed = False

    for linter in linters:
        if not run_command(linter.command, linter.description):
            failed = True

    if failed:
        logging.warning("\nLinting failed!")
        return False

    logging.info("\nAll linters passed!")
    return True

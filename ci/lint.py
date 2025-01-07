"""Linting checks module."""

import subprocess
import sys
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
    print(f"Running {description}...")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"{description} failed:")
        print(result.stdout)
        print(result.stderr)
        return False
    print(f"{description} passed!")
    return True


def get_linters() -> List[LintCommand]:
    """Get all linter commands with fixing enabled where possible."""
    return [
        LintCommand(
            "Black formatter",
            ["black", "--config", "ci/configs/black.toml", "."],
        ),
        LintCommand(
            "isort",
            ["isort", "--settings-path", "ci/configs/isort.toml", "."],
        ),
        LintCommand(
            "mypy type checking",
            ["mypy", "--config-file", "ci/configs/mypy.ini", "."],
        ),
        LintCommand(
            "flake8 linting",
            ["flake8", "--config", "ci/configs/.flake8"],
        ),
    ]


def run_lint() -> None:
    """Run all linters with automatic fixing where possible."""
    print("Running linters...\n")

    linters = get_linters()
    failed = False

    for linter in linters:
        if not run_command(linter.command, linter.description):
            failed = True

    if failed:
        sys.exit(1)

    print("\nAll linters passed!")

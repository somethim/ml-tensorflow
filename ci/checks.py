"""Linting checks module."""

import subprocess
import sys
from dataclasses import dataclass
from typing import List


@dataclass
class LintCommand:
    """Lint command configuration."""

    description: str
    command: List[str]


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
            ["poetry", "run", "black", "."],
        ),
        LintCommand(
            "isort",
            ["poetry", "run", "isort", "."],
        ),
        LintCommand(
            "mypy type checking",
            ["poetry", "run", "mypy", "."],
        ),
        LintCommand(
            "pyright type checking",
            ["poetry", "run", "pyright", "."],
        ),
        LintCommand(
            "flake8 linting",
            ["poetry", "run", "flake8", "--max-line-length=100"],
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

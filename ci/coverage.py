"""Coverage report generation module."""

import subprocess
from typing import List


def run_command(command: List[str]) -> int:
    """Run a command and return its exit code."""
    try:
        subprocess.run(command, check=True)
        return 0
    except subprocess.CalledProcessError as e:
        return e.returncode


def generate_coverage_report() -> None:
    """Generate coverage report."""
    commands = [
        ["pytest", "--cov", "--cov-report=html"],
        ["coverage", "report"],
    ]

    failed = False
    for command in commands:
        if run_command(command) != 0:
            failed = True

    if failed:
        raise SystemExit(1)

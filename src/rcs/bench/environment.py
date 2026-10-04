"""Capture host and package metadata alongside every benchmark result."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkEnvironment:
    """Minimal reproducibility metadata without recording user data."""

    captured_at: str
    platform: str
    python: str
    cpu_count: int | None
    git_commit: str | None
    packages: dict[str, str]


def _git_commit() -> str | None:
    if shutil.which("git") is None:
        return None
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, check=False, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def capture_environment() -> BenchmarkEnvironment:
    """Return portable metadata to embed in a benchmark JSON result."""
    package_names = ("numpy", "onnx", "onnxruntime", "opencv-python-headless")
    packages: dict[str, str] = {}
    for package in package_names:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not-installed"
    return BenchmarkEnvironment(
        captured_at=datetime.now(UTC).isoformat(),
        platform=platform.platform(),
        python=platform.python_version(),
        cpu_count=os.cpu_count(),
        git_commit=_git_commit(),
        packages=packages,
    )


def main() -> None:
    """Write a local environment snapshot; benchmarks will embed the same structure."""
    output = Path("results/environment.json")
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(asdict(capture_environment()), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

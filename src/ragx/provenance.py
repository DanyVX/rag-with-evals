"""Runtime provenance captured with every benchmark artifact."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class EnvironmentProvenance:
    created_at_utc: str
    python_version: str
    platform: str
    processor: str
    git_revision: str | None
    cuda_available: bool


def _git_revision(project_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def collect_environment(project_root: Path) -> dict[str, object]:
    """Return JSON-serializable environment metadata without sensitive values."""
    value = EnvironmentProvenance(
        created_at_utc=datetime.now(UTC).isoformat(),
        python_version=sys.version,
        platform=platform.platform(),
        processor=platform.processor(),
        git_revision=_git_revision(project_root),
        cuda_available=shutil.which("nvidia-smi") is not None,
    )
    return asdict(value)

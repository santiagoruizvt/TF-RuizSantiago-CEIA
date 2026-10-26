"""Procedencia de una corrida: con qué código y en qué entorno se generó.

Permite responder, para cada modelo o resultado de la memoria, de qué commit
salió y si el código tenía cambios sin commitear.
"""

import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _git(*args: str, cwd: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(path: str | Path = PACKAGE_DIR) -> dict[str, Any]:
    """Commit, rama y estado del repositorio que contiene `path`.

    `dirty` es True si hay cambios sin commitear en archivos versionados (los
    archivos sin versionar, como datos o checkpoints, no cuentan). Si no hay git
    o `path` no está en un repositorio, los campos quedan en None.
    """
    path = Path(path)
    commit = _git("rev-parse", "HEAD", cwd=path)
    if commit is None:
        return {"commit": None, "short": None, "branch": None, "describe": None, "dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=path)
    return {
        "commit": commit,
        "short": commit[:7],
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD", cwd=path),
        "describe": _git("describe", "--tags", "--always", "--dirty", cwd=path),
        "dirty": bool(status) if status is not None else None,
    }


def environment_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        info["torch"] = None
    return info


def provenance() -> dict[str, Any]:
    """Git + entorno + momento, listo para guardar en un checkpoint o un JSON."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_info(),
        "environment": environment_info(),
    }

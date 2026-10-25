"""Lectura de configuraciones .yaml y hash para invalidar caches."""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config {path} must be a mapping")
    return cfg


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash estable (independiente del orden de las claves) de una configuración."""
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

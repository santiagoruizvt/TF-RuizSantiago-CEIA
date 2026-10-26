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


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Aplica overrides `clave.anidada=valor` (el valor se interpreta como YAML).

    Ejemplo: ["train.epochs=5", "data.input_size=36"]. Solo se pueden pisar
    claves existentes, para que un error de tipeo no pase inadvertido.
    """
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"Override must be key=value, got {item!r}")
        *parents, leaf = key.split(".")
        node = cfg
        for p in parents:
            node = node[p]
        if leaf not in node:
            raise KeyError(f"Unknown config key: {key}")
        node[leaf] = _parse_value(raw)
    return cfg


def _parse_value(raw: str) -> Any:
    """Interpreta el valor como YAML, aceptando notación científica sin punto.

    PyYAML (YAML 1.1) lee "1e-4" como texto: solo reconoce "1.0e-4". Sin esta
    corrección, `--set train.lr=1e-4` le pasaría un string al optimizador.
    """
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def config_hash(cfg: dict[str, Any]) -> str:
    """Hash estable (independiente del orden de las claves) de una configuración."""
    payload = json.dumps(cfg, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

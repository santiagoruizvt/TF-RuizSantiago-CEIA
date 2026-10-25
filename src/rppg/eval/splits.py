"""Particiones por sujeto (decisión 10.5): test fijo + entrenamiento/validación.

Las particiones se generan una vez con semilla fija y quedan versionadas en
`configs/splits.yaml`. Ambos enfoques se evalúan sobre el mismo test (R5).
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

SPLIT_NAMES = ("train", "val", "test")


def make_splits(
    subjects: Sequence[str], n_test: int, n_val: int, seed: int
) -> dict[str, list[str]]:
    """Reparte sujetos al azar (con semilla) en train/val/test, sin repetir."""
    if n_test + n_val >= len(subjects):
        raise ValueError("n_test + n_val must leave subjects for training")
    order = [str(s) for s in np.random.default_rng(seed).permutation(sorted(subjects))]
    splits = {
        "test": order[:n_test],
        "val": order[n_test : n_test + n_val],
        "train": order[n_test + n_val :],
    }
    return {k: sorted(v, key=_subject_number) for k, v in splits.items()}


def assert_disjoint(splits: dict[str, Sequence[str]]) -> None:
    """Falla si algún sujeto aparece en más de una partición (requerimiento del plan)."""
    seen: dict[str, str] = {}
    for name in SPLIT_NAMES:
        for subject in splits.get(name, []):
            if subject in seen:
                raise ValueError(f"{subject} is in both '{seen[subject]}' and '{name}'")
            seen[subject] = name


def load_splits(path: str | Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    splits = {name: list(cfg[name]) for name in SPLIT_NAMES}
    assert_disjoint(splits)
    return splits


def _subject_number(subject: str) -> int:
    digits = "".join(c for c in subject if c.isdigit())
    return int(digits) if digits else 0

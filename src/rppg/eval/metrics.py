"""Métricas de evaluación comunes a ambos enfoques (sección 6 del plan).

Los pares con NaN en cualquiera de los dos lados (ventanas inválidas) se descartan.
"""

import numpy as np


def _paired(estimate: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    est, ref = np.asarray(estimate, dtype=float), np.asarray(reference, dtype=float)
    if est.shape != ref.shape:
        raise ValueError(f"Shape mismatch: {est.shape} vs {ref.shape}")
    keep = np.isfinite(est) & np.isfinite(ref)
    return est[keep], ref[keep]


def mae(estimate: np.ndarray, reference: np.ndarray) -> float:
    est, ref = _paired(estimate, reference)
    return float(np.mean(np.abs(est - ref)))


def rmse(estimate: np.ndarray, reference: np.ndarray) -> float:
    est, ref = _paired(estimate, reference)
    return float(np.sqrt(np.mean((est - ref) ** 2)))


def pearson(estimate: np.ndarray, reference: np.ndarray) -> float:
    est, ref = _paired(estimate, reference)
    if len(est) < 2 or np.std(est) == 0 or np.std(ref) == 0:
        return float("nan")
    return float(np.corrcoef(est, ref)[0, 1])


def summarize(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    est, _ = _paired(estimate, reference)
    return {
        "mae": mae(estimate, reference),
        "rmse": rmse(estimate, reference),
        "pearson": pearson(estimate, reference),
        "n": int(len(est)),
    }

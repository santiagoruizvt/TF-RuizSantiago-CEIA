"""Fixtures compartidas.

`synthetic_ubfc` genera un mini-dataset con el mismo formato que UBFC-rPPG
(DATASET_2) y una HR conocida por sujeto, para que los tests de integración
corran sin el dataset real (por ejemplo en CI). Reproduce las rarezas
observadas en los archivos reales: timestamps irregulares con un hueco de ~1 s
y una HR del oxímetro con cortes.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

REAL_UBFC_ROOT = Path(__file__).resolve().parents[1] / "data" / "raw"

SYNTHETIC_FPS = 30.0
SYNTHETIC_DURATION_S = 30.0  # deja ventanas de 10 s válidas a ambos lados del hueco
SYNTHETIC_SIZE = (64, 48)  # (ancho, alto): chico para que el test sea rápido
SYNTHETIC_SUBJECTS = {"subject1": 72.0, "subject3": 96.0}  # HR verdadera en bpm


@dataclass
class SyntheticDataset:
    root: Path
    true_hr_bpm: dict[str, float]


def _irregular_timestamps(n: int, fs: float, rng: np.random.Generator) -> np.ndarray:
    """Timestamps con jitter alrededor de 1/fs y un hueco de ~1 s a mitad del video."""
    dt = rng.uniform(0.6, 1.4, size=n - 1) / fs
    dt[n // 2] = 1.0
    return np.concatenate([[0.0], np.cumsum(dt)])


def _write_subject(folder: Path, hr_bpm: float, rng: np.random.Generator) -> None:
    folder.mkdir(parents=True)
    n = int(SYNTHETIC_DURATION_S * SYNTHETIC_FPS)
    t = _irregular_timestamps(n, SYNTHETIC_FPS, rng)
    pulse = np.sin(2 * np.pi * hr_bpm / 60 * t)

    # Video: fondo gris y un parche central cuyo canal verde late con la HR.
    # MJPG es con pérdida, por eso la amplitud del pulso es de varios niveles.
    w, h = SYNTHETIC_SIZE
    writer = cv2.VideoWriter(
        str(folder / "vid.avi"), cv2.VideoWriter_fourcc(*"MJPG"), SYNTHETIC_FPS, (w, h)
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot write MJPG video on this platform")
    for p in pulse:
        frame = np.full((h, w, 3), 90, dtype=np.uint8)  # BGR
        frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = (110, int(round(150 + 6 * p)), 170)
        writer.write(frame)
    writer.release()

    # Ground truth: PPG con ruido, HR del equipo con cortes (~1 bpm) y timestamps.
    ppg = pulse + 0.1 * rng.standard_normal(n)
    hr_device = np.full(n, hr_bpm)
    hr_device[n // 3 : n // 3 + int(2 * SYNTHETIC_FPS)] = 1.0
    np.savetxt(folder / "ground_truth.txt", np.vstack([ppg, hr_device, t]), fmt="%.7e")


@pytest.fixture(scope="session")
def synthetic_ubfc(tmp_path_factory) -> SyntheticDataset:
    root = tmp_path_factory.mktemp("ubfc_synthetic")
    rng = np.random.default_rng(42)
    for subject, hr in SYNTHETIC_SUBJECTS.items():
        _write_subject(root / subject, hr, rng)
    # Carpeta incompleta: el loader debe ignorarla.
    (root / "subject2").mkdir()
    return SyntheticDataset(root=root, true_hr_bpm=dict(SYNTHETIC_SUBJECTS))


@pytest.fixture(scope="session")
def real_ubfc_root() -> Path:
    if not any(REAL_UBFC_ROOT.glob("subject*/ground_truth.txt")):
        pytest.skip("UBFC-rPPG not available locally")
    return REAL_UBFC_ROOT

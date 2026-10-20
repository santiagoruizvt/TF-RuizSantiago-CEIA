"""Loader de UBFC-rPPG (DATASET_2).

Estructura: <root>/subjectN/{vid.avi, ground_truth.txt}. ground_truth.txt tiene
tres filas separadas por espacios, con un valor por frame de video:
    1. PPG de contacto (oxímetro de dedo)
    2. HR que reporta el oxímetro (bpm). Tiene cortes (valores ~1 bpm): NO usarla
       como referencia; derivar la HR desde la PPG.
    3. timestamps de cada frame (s). Irregulares, con huecos de hasta ~1 s.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class GroundTruth:
    ppg: np.ndarray  # (T,)
    hr_device: np.ndarray  # (T,) bpm, poco confiable
    t: np.ndarray  # (T,) segundos


class UBFCrPPG:
    name = "ubfc_rppg"

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

    def subjects(self) -> list[str]:
        """Sujetos que tienen ambos archivos, ordenados numéricamente."""
        dirs = [
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and (d / "vid.avi").is_file() and (d / "ground_truth.txt").is_file()
        ]
        return sorted(dirs, key=lambda s: int(re.sub(r"\D", "", s) or 0))

    def video_path(self, subject: str) -> Path:
        return self.root / subject / "vid.avi"

    def ground_truth(self, subject: str) -> GroundTruth:
        rows = np.loadtxt(self.root / subject / "ground_truth.txt")
        if rows.ndim != 2 or rows.shape[0] != 3:
            raise ValueError(f"Unexpected ground_truth.txt shape {rows.shape} for {subject}")
        return GroundTruth(ppg=rows[0], hr_device=rows[1], t=rows[2])

    def frame_count(self, subject: str) -> int:
        cap = cv2.VideoCapture(str(self.video_path(subject)))
        try:
            return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()

    def frames(self, subject: str) -> Iterator[np.ndarray]:
        """Devuelve frames RGB uint8 (H, W, 3) de a uno (cada video pesa ~1,5 GB sin comprimir)."""
        cap = cv2.VideoCapture(str(self.video_path(subject)))
        if not cap.isOpened():
            raise OSError(f"Cannot open {self.video_path(subject)}")
        try:
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally:
            cap.release()

"""Datos para el DL: recortes de cara (de `scripts/extract_traces.py`) + PPG de referencia.

Cada muestra es un tramo (chunk) de `chunk_length` frames consecutivos:
    x: (3, T, S, S) float32, estandarizado por tramo
    y: (T,) float32, PPG del oxímetro estandarizada por tramo

La PPG de UBFC tiene una muestra por frame, así que recortes y etiqueta están
alineados índice a índice sin remuestrear.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from rppg.data.ubfc import UBFCrPPG


@dataclass
class SubjectClip:
    subject: str
    frames: np.ndarray  # (N, S, S, 3) uint8, ya redimensionados a input_size
    ppg: np.ndarray  # (N,) float
    t: np.ndarray  # (N,) segundos
    detected: np.ndarray  # (N,) bool


def load_subject(
    subject: str, processed_root: str | Path, raw_root: str | Path, input_size: int
) -> SubjectClip:
    with np.load(Path(processed_root) / subject / "crops.npz") as z:
        crops, detected, t = z["crops"], z["detected"], z["t"]
    gt = UBFCrPPG(raw_root).ground_truth(subject)
    if len(gt.ppg) != len(crops):
        raise ValueError(f"{subject}: {len(crops)} crops but {len(gt.ppg)} PPG samples")
    if crops.shape[1] != input_size:
        crops = np.stack(
            [cv2.resize(f, (input_size, input_size), interpolation=cv2.INTER_AREA) for f in crops]
        )
    return SubjectClip(subject, crops, gt.ppg.astype(np.float32), t, detected)


def standardize_frames(frames: np.ndarray) -> np.ndarray:
    """(T, S, S, 3) uint8 -> (3, T, S, S) float32 con media 0 y desvío 1 por canal.

    La estandarización por tramo elimina el nivel absoluto de brillo y contraste
    de cada video: la red solo ve variaciones dentro del tramo.
    """
    x = frames.astype(np.float32) / 255.0
    # Estadísticos en float64: en float32 el error de acumulación se nota en
    # tramos de poco contraste, después de dividir por un desvío chico.
    mean = x.mean(axis=(0, 1, 2), keepdims=True, dtype=np.float64)
    std = x.std(axis=(0, 1, 2), keepdims=True, dtype=np.float64) + 1e-6
    return np.ascontiguousarray(((x - mean) / std).astype(np.float32).transpose(3, 0, 1, 2))


def interpolate_frames(frames: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Frames en posiciones temporales fraccionarias, por interpolación lineal.

    frames: (N, S, S, 3) uint8; positions: (T,) índices en [0, N - 1].
    Devuelve (T, S, S, 3) float32.
    """
    i0 = np.clip(np.floor(positions).astype(int), 0, len(frames) - 1)
    i1 = np.clip(i0 + 1, 0, len(frames) - 1)
    w = (positions - i0).astype(np.float32)[:, None, None, None]
    return (1 - w) * frames[i0].astype(np.float32) + w * frames[i1].astype(np.float32)


def standardize_signal(y: np.ndarray) -> np.ndarray:
    return ((y - y.mean()) / (y.std() + 1e-6)).astype(np.float32)


class Augmenter:
    """Aumento de datos para el entrenamiento (mitigación de R1 y R4).

    Un cambio de brillo o contraste constante en todo el tramo no tendría efecto,
    porque la estandarización lo elimina. Por eso se aplican:
    - espejado horizontal;
    - deriva lenta de iluminación: una ganancia que varía linealmente a lo largo
      del tramo, como una luz que cambia durante la videollamada;
    - ruido gaussiano por píxel, como el ruido del sensor de una webcam;
    - velocidad temporal (ver `speed_factor`): lo aplica `ChunkDataset`, porque
      cambia a la vez el video y la etiqueta.
    """

    def __init__(self, cfg: Mapping[str, Any], seed: int) -> None:
        self.flip_p = cfg.get("flip_p", 0.0)
        self.gain_drift = cfg.get("gain_drift", 0.0)
        self.noise_std = cfg.get("noise_std", 0.0)
        self.speed_p = cfg.get("speed_p", 0.0)
        self.speed_range = tuple(cfg.get("speed_range", (1.0, 1.0)))
        if not 0 < self.speed_range[0] <= self.speed_range[1]:
            raise ValueError(f"Invalid speed_range {self.speed_range}")
        self.rng = np.random.default_rng(seed)

    def speed_factor(self) -> float:
        """Factor de velocidad para el próximo tramo (1.0 = sin cambio).

        Reproducir un tramo s veces más rápido multiplica su HR por s. Con un
        rango de 0,6 a 1,4, sujetos con HR de 77 a 129 bpm (los de train)
        aportan ejemplos de ~46 a ~180 bpm.
        """
        if self.speed_p and self.rng.random() < self.speed_p:
            return float(self.rng.uniform(*self.speed_range))
        return 1.0

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        x = frames.astype(np.float32)
        if self.rng.random() < self.flip_p:
            x = x[:, :, ::-1]
        if self.gain_drift:
            start, end = 1 + self.rng.uniform(-self.gain_drift, self.gain_drift, size=2)
            x = x * np.linspace(start, end, len(x), dtype=np.float32)[:, None, None, None]
        if self.noise_std:
            x = x + self.rng.normal(0, self.noise_std * 255, size=x.shape).astype(np.float32)
        return np.clip(x, 0, 255)


class ChunkDataset(Dataset):
    """Tramos de `chunk_length` frames con paso `stride` sobre varios sujetos.

    Se descartan los tramos que contienen frames sin detección facial.
    """

    def __init__(
        self,
        clips: Sequence[SubjectClip],
        chunk_length: int,
        stride: int,
        augmenter: Augmenter | None = None,
    ) -> None:
        self.clips = list(clips)
        self.chunk_length = chunk_length
        self.augmenter = augmenter
        self.index: list[tuple[int, int]] = []
        for c, clip in enumerate(self.clips):
            for start in range(0, len(clip.frames) - chunk_length + 1, stride):
                if clip.detected[start : start + chunk_length].all():
                    self.index.append((c, start))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        c, start = self.index[i]
        clip = self.clips[c]
        speed = self.augmenter.speed_factor() if self.augmenter is not None else 1.0
        frames, ppg = self._window(clip, start, speed)
        if self.augmenter is not None:
            frames = self.augmenter(frames)
        x = standardize_frames(frames)
        y = standardize_signal(ppg)
        return torch.from_numpy(x), torch.from_numpy(y)

    def _window(self, clip: SubjectClip, start: int, speed: float) -> tuple[np.ndarray, np.ndarray]:
        """Tramo de `chunk_length` frames y su PPG, reproducido a `speed` veces.

        Con speed != 1 se toman round(chunk_length * speed) frames de origen y
        se remuestrean por interpolación lineal a chunk_length, igual que la
        PPG. Si el tramo de origen no entra en el video o tiene frames sin
        detección, se usa el tramo original (speed = 1).
        """
        n, length = len(clip.frames), self.chunk_length
        src_len = int(round(length * speed))
        if speed != 1.0 and src_len <= n:
            src_start = min(start, n - src_len)
            if clip.detected[src_start : src_start + src_len].all():
                pos = src_start + np.linspace(0, src_len - 1, length)
                ppg = np.interp(pos, np.arange(n), clip.ppg).astype(np.float32)
                return interpolate_frames(clip.frames, pos), ppg
        return clip.frames[start : start + length], clip.ppg[start : start + length]

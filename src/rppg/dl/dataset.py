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
from rppg.dsp.spectral import dominant_frequency


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
    - velocidad temporal (ver `speed_factor`) y recorte temporal aleatorio (ver
      `start_offset`): los aplica `ChunkDataset`, porque cambian a la vez el
      video y la etiqueta.

    Toda transformación que toca la PPG toca también el video: transformar solo
    la etiqueta enseñaría una correspondencia falsa.
    """

    SPEED_MODES = ("uniform", "hr_balanced")

    def __init__(self, cfg: Mapping[str, Any], seed: int) -> None:
        self.flip_p = cfg.get("flip_p", 0.0)
        self.gain_drift = cfg.get("gain_drift", 0.0)
        self.noise_std = cfg.get("noise_std", 0.0)
        self.speed_p = cfg.get("speed_p", 0.0)
        self.speed_range = tuple(cfg.get("speed_range", (1.0, 1.0)))
        if not 0 < self.speed_range[0] <= self.speed_range[1]:
            raise ValueError(f"Invalid speed_range {self.speed_range}")
        self.speed_mode = cfg.get("speed_mode", "uniform")
        if self.speed_mode not in self.SPEED_MODES:
            raise ValueError(f"speed_mode must be one of {self.SPEED_MODES}")
        self.target_hr_range = tuple(cfg.get("target_hr_range", (45.0, 150.0)))
        if not 0 < self.target_hr_range[0] <= self.target_hr_range[1]:
            raise ValueError(f"Invalid target_hr_range {self.target_hr_range}")
        self.temporal_jitter = bool(cfg.get("temporal_jitter", False))
        self.rng = np.random.default_rng(seed)

    @property
    def needs_chunk_hr(self) -> bool:
        return bool(self.speed_p) and self.speed_mode == "hr_balanced"

    def speed_factor(self, chunk_hr: float | None = None) -> float:
        """Factor de velocidad para el próximo tramo (1.0 = sin cambio).

        Reproducir un tramo s veces más rápido multiplica su HR por s.
        - `uniform`: s uniforme en `speed_range`. Con 0,6 a 1,4, sujetos con HR
          de 77 a 129 bpm aportan ejemplos de ~46 a ~180 bpm, pero la
          distribución sigue concentrada donde están los sujetos.
        - `hr_balanced` (sobremuestreo por HR): se elige una HR objetivo
          uniforme en `target_hr_range` y s = objetivo / HR del tramo, recortado
          a `speed_range`. Así el modelo ve tantos tramos de 50 bpm como de 100.
        """
        if not (self.speed_p and self.rng.random() < self.speed_p):
            return 1.0
        if self.speed_mode == "uniform":
            return float(self.rng.uniform(*self.speed_range))
        if chunk_hr is None or not np.isfinite(chunk_hr) or chunk_hr <= 0:
            return 1.0
        target = self.rng.uniform(*self.target_hr_range)
        return float(np.clip(target / chunk_hr, *self.speed_range))

    def start_offset(self, max_offset: int) -> int:
        """Desplazamiento aleatorio del inicio del tramo, en [0, max_offset).

        Sin esto, los tramos de entrenamiento arrancan siempre en la misma
        grilla (múltiplos de `train_stride`).
        """
        if not self.temporal_jitter or max_offset <= 1:
            return 0
        return int(self.rng.integers(0, max_offset))

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


def chunk_hr(
    clip: SubjectClip,
    start: int,
    length: int,
    band_hz: tuple[float, float],
    min_window_s: float = 10.0,
) -> float:
    """HR (bpm) de la PPG de referencia alrededor de un tramo.

    Un tramo de 128 frames dura ~4,3 s, y su resolución espectral (~14 bpm) es
    gruesa. Se usa una ventana de al menos `min_window_s`, centrada en el tramo,
    con el estimador del protocolo (periodograma con Hann, zero-padding e
    interpolación parabólica). La frecuencia de muestreo sale de los timestamps
    de esa ventana. Solo se usa para elegir la velocidad del aumento.
    """
    n = len(clip.ppg)
    fs_clip = (n - 1) / (clip.t[-1] - clip.t[0])
    win = min(n, max(length, int(round(min_window_s * fs_clip))))
    lo = max(0, min(start + length // 2 - win // 2, n - win))
    hi = lo + win
    fs = (win - 1) / (clip.t[hi - 1] - clip.t[lo])
    f, _ = dominant_frequency(clip.ppg[lo:hi], fs, band_hz, nfft=4096)
    return 60.0 * f


class ChunkDataset(Dataset):
    """Tramos de `chunk_length` frames con paso `stride` sobre varios sujetos.

    Se descartan los tramos que contienen frames sin detección facial. Con
    sobremuestreo por HR (`Augmenter.speed_mode = "hr_balanced"`), la HR de cada
    tramo se calcula una vez al construir el dataset, en la banda `hr_band_hz`.
    """

    def __init__(
        self,
        clips: Sequence[SubjectClip],
        chunk_length: int,
        stride: int,
        augmenter: Augmenter | None = None,
        hr_band_hz: tuple[float, float] = (0.7, 4.0),
    ) -> None:
        self.clips = list(clips)
        self.chunk_length = chunk_length
        self.stride = stride
        self.augmenter = augmenter
        self.index: list[tuple[int, int]] = []
        for c, clip in enumerate(self.clips):
            for start in range(0, len(clip.frames) - chunk_length + 1, stride):
                if clip.detected[start : start + chunk_length].all():
                    self.index.append((c, start))
        self.hr: np.ndarray | None = None
        if augmenter is not None and augmenter.needs_chunk_hr:
            self.hr = np.array(
                [chunk_hr(self.clips[c], s, chunk_length, hr_band_hz) for c, s in self.index]
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        c, start = self.index[i]
        clip = self.clips[c]
        speed = 1.0
        if self.augmenter is not None:
            start = self._jittered_start(clip, start)
            speed = self.augmenter.speed_factor(None if self.hr is None else self.hr[i])
        frames, ppg = self._window(clip, start, speed)
        if self.augmenter is not None:
            frames = self.augmenter(frames)
        x = standardize_frames(frames)
        y = standardize_signal(ppg)
        return torch.from_numpy(x), torch.from_numpy(y)

    def _jittered_start(self, clip: SubjectClip, start: int) -> int:
        """Inicio desplazado al azar dentro de [start, start + stride).

        Si el tramo desplazado se sale del video o tiene frames sin detección,
        se usa el inicio original.
        """
        offset = self.augmenter.start_offset(self.stride)
        shifted = start + offset
        if offset and shifted + self.chunk_length <= len(clip.frames):
            if clip.detected[shifted : shifted + self.chunk_length].all():
                return shifted
        return start

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

"""Inferencia de PhysNet sobre un video completo."""

import numpy as np
import torch
from torch import nn

from rppg.dl.dataset import standardize_frames


def chunk_starts(n: int, chunk_length: int, stride: int) -> list[int]:
    """Inicios de los tramos que cubren los N frames.

    Los tramos avanzan de a `stride`. Si el último no llega al final del video,
    se agrega uno alineado al final.
    """
    if n < chunk_length:
        raise ValueError(f"Video has {n} frames, fewer than chunk_length={chunk_length}")
    if not 0 < stride <= chunk_length:
        raise ValueError(f"stride must be in (0, {chunk_length}], got {stride}")
    starts = list(range(0, n - chunk_length + 1, stride))
    if starts[-1] + chunk_length < n:
        starts.append(n - chunk_length)
    return starts


def overlap_weights(chunk_length: int) -> np.ndarray:
    """Ventana de Hann sin ceros en los extremos, para el promedio ponderado.

    Pesa menos los bordes de cada tramo, donde la red tiene menos contexto
    temporal y la estandarización por tramo produce saltos. Sin ceros, cada
    frame tiene peso positivo aunque lo cubra un solo tramo.
    """
    return np.hanning(chunk_length + 2)[1:-1]


@torch.no_grad()
def _predict_chunks(
    model: nn.Module,
    frames: np.ndarray,
    starts: list[int],
    chunk_length: int,
    device: str | torch.device,
    batch_size: int,
    flip: bool,
) -> np.ndarray:
    """Predicción estandarizada (media 0, desvío 1) de cada tramo: (n_tramos, L)."""
    preds = []
    for b in range(0, len(starts), batch_size):
        x = np.stack(
            [standardize_frames(frames[s : s + chunk_length]) for s in starts[b : b + batch_size]]
        )
        if flip:
            x = np.ascontiguousarray(x[..., ::-1])  # espejado horizontal (eje W)
        preds.append(model(torch.from_numpy(x).to(device)).float().cpu().numpy())
    p = np.concatenate(preds)
    # La escala de cada tramo es arbitraria: se estandariza antes de combinar.
    return (p - p.mean(axis=1, keepdims=True)) / (p.std(axis=1, keepdims=True) + 1e-6)


def predict_signal(
    model: nn.Module,
    frames: np.ndarray,
    chunk_length: int,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
    stride: int | None = None,
    flip_average: bool = False,
) -> np.ndarray:
    """Señal rPPG (N,) para un video de N frames (N, S, S, 3) uint8.

    Cada tramo se estandariza igual que en el entrenamiento, y su salida se
    estandariza antes de combinar, porque la escala es arbitraria.

    - `stride=None` (o igual a `chunk_length`): tramos sin solapamiento. El
      último se alinea al final del video y de él solo se toman los frames que
      faltaban. Es el modo que usa el entrenamiento para elegir la época.
    - `stride < chunk_length`: tramos solapados, combinados por promedio
      ponderado (*overlap-add*) con `overlap_weights`. Cada frame queda como
      promedio de varias predicciones, lo que reduce el ruido y suaviza los
      saltos entre tramos.
    - `flip_average=True`: promedia además con la predicción sobre el video
      espejado horizontalmente (el mismo espejado que se usa al entrenar).
    """
    n = len(frames)
    stride = chunk_length if stride is None else stride
    starts = chunk_starts(n, chunk_length, stride)
    model.eval()
    preds = _predict_chunks(model, frames, starts, chunk_length, device, batch_size, flip=False)
    if flip_average:
        flipped = _predict_chunks(model, frames, starts, chunk_length, device, batch_size, True)
        preds = (preds + flipped) / 2

    if stride == chunk_length:
        out = np.empty(n, dtype=np.float32)
        covered_until = 0
        for s, p in zip(starts, preds, strict=True):
            out[covered_until : s + chunk_length] = p[covered_until - s :]
            covered_until = s + chunk_length
        return out

    w = overlap_weights(chunk_length)
    acc = np.zeros(n)
    weight = np.zeros(n)
    for s, p in zip(starts, preds, strict=True):
        acc[s : s + chunk_length] += w * p
        weight[s : s + chunk_length] += w
    return (acc / weight).astype(np.float32)

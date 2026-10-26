"""Inferencia de PhysNet sobre un video completo."""

import numpy as np
import torch
from torch import nn

from rppg.dl.dataset import standardize_frames


@torch.no_grad()
def predict_signal(
    model: nn.Module,
    frames: np.ndarray,
    chunk_length: int,
    device: str | torch.device = "cpu",
    batch_size: int = 4,
) -> np.ndarray:
    """Señal rPPG (N,) para un video de N frames (N, S, S, 3) uint8.

    Recorre el video en tramos sin solapamiento. El último tramo se alinea al
    final del video y de él solo se toman los frames que faltaban, así la salida
    cubre exactamente los N frames. Cada tramo se estandariza igual que en el
    entrenamiento. Como la escala de cada tramo es arbitraria, la salida se
    estandariza por tramo antes de concatenar.
    """
    n = len(frames)
    if n < chunk_length:
        raise ValueError(f"Video has {n} frames, fewer than chunk_length={chunk_length}")
    # (inicio del tramo, primer frame de la salida que aporta ese tramo)
    chunks = [(s, s) for s in range(0, n - chunk_length + 1, chunk_length)]
    covered_until = chunks[-1][0] + chunk_length
    if covered_until < n:
        chunks.append((n - chunk_length, covered_until))

    model.eval()
    out = np.empty(n, dtype=np.float32)
    for b in range(0, len(chunks), batch_size):
        batch = chunks[b : b + batch_size]
        x = np.stack([standardize_frames(frames[s : s + chunk_length]) for s, _ in batch])
        pred = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
        for (s, first), p in zip(batch, pred, strict=True):
            p = (p - p.mean()) / (p.std() + 1e-6)
            out[first : s + chunk_length] = p[first - s :]
    return out

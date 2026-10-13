"""Remuestreo de señales con timestamps irregulares a una grilla uniforme."""

from dataclasses import dataclass

import numpy as np


@dataclass
class UniformSignal:
    t: np.ndarray  # (N,) segundos, uniforme con paso 1/fs
    x: np.ndarray  # (N,) o (N, C)
    valid: np.ndarray  # (N,) bool: False si la muestra cae en un hueco mayor a max_gap_s
    fs: float


def resample_uniform(
    t: np.ndarray, x: np.ndarray, fs: float, max_gap_s: float = 0.5
) -> UniformSignal:
    """Interpola linealmente x(t) sobre una grilla uniforme a fs.

    Las muestras con NaN (p. ej. frames sin detección facial) se descartan antes
    de interpolar. Los timestamps repetidos se promedian. Las muestras de salida
    que caen dentro de un hueco entre entradas válidas consecutivas mayor a
    max_gap_s se marcan inválidas (igual se interpolan, para que el arreglo
    quede contiguo).
    """
    t = np.asarray(t, dtype=float)
    x = np.asarray(x, dtype=float)
    squeeze = x.ndim == 1
    x2 = x[:, None] if squeeze else x.reshape(len(x), -1)
    if len(t) != len(x2):
        raise ValueError("t and x must have the same length")

    keep = np.all(np.isfinite(x2), axis=1) & np.isfinite(t)
    t, x2 = t[keep], x2[keep]
    order = np.argsort(t, kind="stable")
    t, x2 = t[order], x2[order]
    t_unique, inverse = np.unique(t, return_inverse=True)
    if len(t_unique) < len(t):
        sums = np.zeros((len(t_unique), x2.shape[1]))
        np.add.at(sums, inverse, x2)
        x2 = sums / np.bincount(inverse)[:, None]
        t = t_unique
    if len(t) < 2:
        raise ValueError("Need at least two valid samples to resample")

    t_out = t[0] + np.arange(int(np.floor((t[-1] - t[0]) * fs)) + 1) / fs
    x_out = np.column_stack([np.interp(t_out, t, x2[:, c]) for c in range(x2.shape[1])])

    # Largo del intervalo de entrada que contiene cada muestra de salida. Las
    # muestras que coinciden con un timestamp de entrada son datos reales, no
    # interpolados.
    idx = np.clip(np.searchsorted(t, t_out, side="right"), 1, len(t) - 1)
    left, right = t[idx - 1], t[idx]
    on_sample = (t_out - left < 1e-6) | (right - t_out < 1e-6)
    valid = (right - left <= max_gap_s) | on_sample

    x_out = x_out[:, 0] if squeeze else x_out.reshape((len(t_out),) + x.shape[1:])
    return UniformSignal(t=t_out, x=x_out, valid=valid, fs=fs)

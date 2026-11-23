"""Pipeline clásico: trazas RGB por ROI (cache) -> señal de pulso -> HR por ventana.

Todo lo que viene después de la señal de pulso es el protocolo común
(`rppg.eval.protocol`), el mismo que se aplica a PhysNet y a la PPG de
referencia.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from rppg.classical.methods import get_method
from rppg.dsp.resample import resample_uniform
from rppg.eval.protocol import compare_hr
from rppg.face.extract import FaceTrack


def combine_rois(
    rgb: np.ndarray, n_pixels: np.ndarray, rois: list[int] | None = None
) -> np.ndarray:
    """Une las ROIs en una traza (T, 3), promediando por cantidad de píxeles.

    Equivale a la media sobre la unión de las ROIs: una mejilla grande pesa más
    que una frente chica. Los frames sin detección (NaN en todas las ROIs) quedan
    en NaN. `rois` selecciona un subconjunto por índice.
    """
    if rois is not None:
        rgb, n_pixels = rgb[:, rois], n_pixels[:, rois]
    weights = np.where(np.isfinite(rgb).all(axis=2), n_pixels, 0).astype(float)
    total = weights.sum(axis=1)
    summed = np.nansum(rgb * weights[:, :, None], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = summed / total[:, None]
    out[total == 0] = np.nan
    return out


def detection_validity(
    t: np.ndarray, detected: np.ndarray, t_grid: np.ndarray, max_gap_s: float
) -> np.ndarray:
    """Validez de cada muestra de `t_grid` según la detección facial.

    Una muestra es inválida si cae en un tramo sin cara mayor a `max_gap_s`
    (entre dos frames detectados consecutivos), o antes del primer frame
    detectado o después del último. Es el mismo criterio de huecos del
    remuestreo (`rppg.dsp.resample`).
    """
    t_det = t[detected]
    if len(t_det) == 0:
        return np.zeros(len(t_grid), dtype=bool)
    inside = (t_grid >= t_det[0] - 1e-6) & (t_grid <= t_det[-1] + 1e-6)
    idx = np.clip(np.searchsorted(t_det, t_grid, side="right"), 1, max(1, len(t_det) - 1))
    if len(t_det) == 1:
        return inside & (np.abs(t_grid - t_det[0]) < 1e-6)
    left, right = t_det[idx - 1], t_det[idx]
    on_sample = (np.abs(t_grid - left) < 1e-6) | (np.abs(right - t_grid) < 1e-6)
    return inside & ((right - left <= max_gap_s) | on_sample)


def pulse_from_track(
    track: FaceTrack,
    method: str,
    fs: float,
    max_gap_s: float,
    params: Mapping[str, Any] | None = None,
    rois: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Señal de pulso de un video sobre una grilla uniforme a `fs`, y su validez.

    La grilla empieza en el primer timestamp del video, como la de la
    referencia, así las ventanas quedan alineadas aunque falte la cara al
    principio. Los frames sin cara se rellenan por interpolación (y repitiendo
    el valor en los extremos) para que el método reciba una señal contigua; la
    validez devuelta marca las muestras que caen en huecos largos o fuera del
    rango detectado.
    """
    roi_idx = None if rois is None else [track.roi_names.index(r) for r in rois]
    rgb = combine_rois(track.rgb, track.n_pixels, roi_idx)
    good = np.isfinite(rgb).all(axis=1)
    if good.sum() < 2:
        raise ValueError("Need at least two frames with a detected face")
    filled = np.column_stack(
        [np.interp(track.t, track.t[good], rgb[good, c]) for c in range(rgb.shape[1])]
    )
    u = resample_uniform(track.t, filled, fs, max_gap_s=max_gap_s)
    valid = u.valid & detection_validity(track.t, good, u.t, max_gap_s)
    pulse = get_method(method)(u.x, fs, **(params or {}))
    return u.t, pulse, valid


def evaluate_subject(
    subject: str,
    track: FaceTrack,
    t_ref: np.ndarray,
    ppg_ref: np.ndarray,
    method: str,
    eval_cfg: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
    rois: list[str] | None = None,
) -> pd.DataFrame:
    """HR estimada vs. referencia, ventana a ventana, para un sujeto y un método."""
    t, pulse, valid = pulse_from_track(
        track, method, eval_cfg["fs"], eval_cfg["max_gap_s"], params=params, rois=rois
    )
    windows = compare_hr(subject, t, pulse, t_ref, ppg_ref, eval_cfg, est_valid=valid)
    windows.insert(0, "method", method)
    return windows

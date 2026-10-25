"""ROIs de piel definidas por landmarks y recortes de cara para el DL."""

from collections.abc import Mapping, Sequence

import cv2
import numpy as np


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    """Máscara booleana (H, W) del polígono (N, 2) en píxeles (x, y)."""
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def skin_mask(
    frame_rgb: np.ndarray, cr_range: Sequence[int], cb_range: Sequence[int]
) -> np.ndarray:
    """Máscara de piel por umbral en YCrCb (Cr y Cb dentro de los rangos)."""
    ycrcb = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2YCrCb)
    cr, cb = ycrcb[..., 1], ycrcb[..., 2]
    return (cr >= cr_range[0]) & (cr <= cr_range[1]) & (cb >= cb_range[0]) & (cb <= cb_range[1])


def roi_means(
    frame_rgb: np.ndarray,
    landmarks: np.ndarray,
    rois: Mapping[str, Sequence[int]],
    skin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Color medio RGB y cantidad de píxeles de cada ROI.

    Devuelve means (R, 3) y n_pixels (R,), en el orden de `rois`. Una ROI sin
    píxeles (fuera de cuadro o vacía tras la máscara de piel) da NaN.
    Trabaja sobre la caja que contiene cada polígono, no sobre el frame entero.
    """
    h, w = frame_rgb.shape[:2]
    means = np.full((len(rois), 3), np.nan)
    counts = np.zeros(len(rois), dtype=np.int64)
    for r, idx in enumerate(rois.values()):
        poly = landmarks[list(idx)]
        x0, y0 = np.floor(poly.min(axis=0)).astype(int)
        x1, y1 = np.ceil(poly.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if x1 <= x0 or y1 <= y0:
            continue
        mask = polygon_mask((y1 - y0, x1 - x0), poly - [x0, y0])
        if skin is not None:
            mask &= skin[y0:y1, x0:x1]
        n = int(mask.sum())
        if n:
            means[r] = frame_rgb[y0:y1, x0:x1][mask].mean(axis=0)
            counts[r] = n
    return means, counts


def face_crop(frame_rgb: np.ndarray, landmarks: np.ndarray, size: int, scale: float) -> np.ndarray:
    """Recorte cuadrado centrado en la cara, redimensionado a (size, size, 3).

    El lado es scale veces el lado mayor de la caja de landmarks. Lo que cae
    fuera del frame se rellena con el borde replicado, así la cara no se
    deforma cerca de los bordes.
    """
    (x_min, y_min), (x_max, y_max) = landmarks.min(axis=0), landmarks.max(axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    half = scale * max(x_max - x_min, y_max - y_min) / 2
    x0, y0 = int(round(cx - half)), int(round(cy - half))
    side = max(int(round(2 * half)), 1)
    h, w = frame_rgb.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    if pad:
        frame_rgb = cv2.copyMakeBorder(frame_rgb, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad, y0 + pad
    crop = frame_rgb[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

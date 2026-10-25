"""Extracción por video: trazas RGB por ROI (clásico) y recortes de cara (DL).

Es la única etapa que recorre píxeles. Se ejecuta una vez por video y el
resultado se guarda en cache (`data/processed/<dataset>/<subject>/`):

- traces.npz: rgb (T, R, 3), n_pixels (T, R), detected (T,), t (T,), roi_names
- crops.npz:  crops (T, S, S, 3) uint8, detected (T,), t (T,)

Los frames sin detección quedan con NaN en rgb y con el último recorte válido
en crops (o ceros si todavía no hubo ninguno); `detected` indica cuáles son.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rppg.face.roi import face_crop, roi_means, skin_mask

Detector = Callable[[np.ndarray, float], np.ndarray | None]


@dataclass
class FaceTrack:
    t: np.ndarray  # (T,) segundos
    detected: np.ndarray  # (T,) bool
    rgb: np.ndarray  # (T, R, 3) float, NaN sin detección
    n_pixels: np.ndarray  # (T, R) int
    roi_names: list[str]
    crops: np.ndarray | None  # (T, S, S, 3) uint8, o None si no se pidieron

    @property
    def detection_rate(self) -> float:
        return float(self.detected.mean()) if len(self.detected) else 0.0


def extract_video(
    frames: Iterable[np.ndarray],
    timestamps: Sequence[float],
    detector: Detector,
    rois: Mapping[str, Sequence[int]],
    skin_ranges: tuple[Sequence[int], Sequence[int]] | None = None,
    crop_size: int | None = None,
    crop_scale: float = 1.5,
) -> FaceTrack:
    """Recorre un video y devuelve sus trazas por ROI y, opcionalmente, recortes.

    skin_ranges = (cr_range, cb_range) activa la máscara de piel en YCrCb.
    crop_size = None desactiva los recortes.
    """
    t = np.asarray(timestamps, dtype=float)
    n = len(t)
    rgb = np.full((n, len(rois), 3), np.nan)
    n_pixels = np.zeros((n, len(rois)), dtype=np.int64)
    detected = np.zeros(n, dtype=bool)
    crops = np.zeros((n, crop_size, crop_size, 3), dtype=np.uint8) if crop_size else None
    last_crop = None

    i = -1
    for i, frame in enumerate(frames):
        if i >= n:
            raise ValueError(f"Video has more frames than timestamps ({n})")
        landmarks = detector(frame, t[i])
        if landmarks is not None:
            detected[i] = True
            skin = skin_mask(frame, *skin_ranges) if skin_ranges else None
            rgb[i], n_pixels[i] = roi_means(frame, landmarks, rois, skin)
            if crops is not None:
                last_crop = face_crop(frame, landmarks, crop_size, crop_scale)
        if crops is not None and last_crop is not None:
            crops[i] = last_crop
    if i + 1 != n:
        raise ValueError(f"Video has {i + 1} frames but {n} timestamps")

    return FaceTrack(
        t=t, detected=detected, rgb=rgb, n_pixels=n_pixels, roi_names=list(rois), crops=crops
    )


def save_track(track: FaceTrack, folder: str | Path, cfg_hash: str) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(
        folder / "traces.npz",
        t=track.t,
        detected=track.detected,
        rgb=track.rgb,
        n_pixels=track.n_pixels,
        roi_names=np.array(track.roi_names),
        config_hash=cfg_hash,
    )
    if track.crops is not None:
        np.savez_compressed(
            folder / "crops.npz",
            t=track.t,
            detected=track.detected,
            crops=track.crops,
            config_hash=cfg_hash,
        )


def load_traces(folder: str | Path) -> FaceTrack:
    """Carga traces.npz (sin los recortes, que son pesados y solo los usa el DL)."""
    with np.load(Path(folder) / "traces.npz") as z:
        return FaceTrack(
            t=z["t"],
            detected=z["detected"],
            rgb=z["rgb"],
            n_pixels=z["n_pixels"],
            roi_names=[str(s) for s in z["roi_names"]],
            crops=None,
        )


def cached_hash(folder: str | Path) -> str | None:
    """Hash de configuración con que se generó la cache, o None si no existe."""
    path = Path(folder) / "traces.npz"
    if not path.is_file():
        return None
    with np.load(path) as z:
        return str(z["config_hash"])

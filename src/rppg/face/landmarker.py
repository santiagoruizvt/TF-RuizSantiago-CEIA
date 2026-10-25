"""Detección de landmarks faciales con MediaPipe Face Landmarker (decisión D1)."""

import hashlib
import urllib.request
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

N_LANDMARKS = 478


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(path: str | Path, url: str, sha256: str) -> Path:
    """Descarga el modelo si falta y verifica su hash.

    El modelo no se versiona en git: se baja una vez a `models/`.
    """
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(path)
    actual = _sha256(path)
    if actual != sha256:
        raise ValueError(f"Model checksum mismatch for {path}: {actual} != {sha256}")
    return path


class MonotonicClock:
    """Convierte timestamps en segundos a milisegundos estrictamente crecientes.

    El modo VIDEO de MediaPipe rechaza timestamps repetidos o que retroceden, y
    UBFC-rPPG tiene timestamps repetidos (intervalos de 0 ms entre frames).
    """

    def __init__(self) -> None:
        self._last_ms = -1

    def __call__(self, t_s: float) -> int:
        ms = max(self._last_ms + 1, int(round(t_s * 1000)))
        self._last_ms = ms
        return ms


class LandmarkSmoother:
    """Media móvil exponencial sobre los landmarks para reducir el jitter de las ROIs.

    alpha es el peso de la muestra nueva (1.0 = sin suavizado). Cuando un frame
    no tiene detección, el estado se reinicia: no se arrastran posiciones viejas
    a través de un hueco.
    """

    def __init__(self, alpha: float) -> None:
        if not 0 < alpha <= 1:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def __call__(self, landmarks: np.ndarray | None) -> np.ndarray | None:
        if landmarks is None:
            self._state = None
            return None
        if self._state is None:
            self._state = landmarks.astype(float)
        else:
            self._state = self.alpha * landmarks + (1 - self.alpha) * self._state
        return self._state.copy()


class VideoLandmarker:
    """Landmarks de una cara por frame, para un único video.

    Usar una instancia nueva por video: MediaPipe exige timestamps crecientes
    dentro de la misma instancia.
    """

    def __init__(
        self,
        model_path: str | Path,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        smoothing: float = 1.0,
    ) -> None:
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._clock = MonotonicClock()
        self._smoother = LandmarkSmoother(smoothing)

    def __call__(self, frame_rgb: np.ndarray, t_s: float) -> np.ndarray | None:
        """Landmarks (478, 2) en píxeles (x, y), o None si no hay cara."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame_rgb))
        result = self._landmarker.detect_for_video(image, self._clock(t_s))
        if not result.face_landmarks:
            return self._smoother(None)
        h, w = frame_rgb.shape[:2]
        xy = np.array([[p.x * w, p.y * h] for p in result.face_landmarks[0]])
        return self._smoother(xy)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "VideoLandmarker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

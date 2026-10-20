"""Filtrado de fase cero y remoción de tendencia. Compartido por los pipelines clásico y de DL."""

import numpy as np
from scipy import sparse
from scipy.signal import butter, detrend, sosfiltfilt
from scipy.sparse.linalg import spsolve


def bandpass(x: np.ndarray, fs: float, low_hz: float, high_hz: float, order: int = 3) -> np.ndarray:
    """Pasa-banda Butterworth aplicado hacia adelante y hacia atrás (fase cero) sobre el eje 0.

    Solo para procesamiento offline: filtfilt no es causal.
    """
    if not 0 < low_hz < high_hz < fs / 2:
        raise ValueError(f"Invalid band [{low_hz}, {high_hz}] Hz for fs={fs} Hz")
    sos = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=0)


def detrend_linear(x: np.ndarray) -> np.ndarray:
    return detrend(x, axis=0, type="linear")


def detrend_smoothness_priors(x: np.ndarray, lam: float = 100.0) -> np.ndarray:
    """Remoción de tendencia por *smoothness priors* (Tarvainen et al., 2002) para una señal 1-D.

    Resta la tendencia z_trend = (I + lam^2 D2^T D2)^-1 z, donde D2 es el
    operador de diferencias de segundo orden. Cuanto mayor es lam, más lenta es
    la tendencia que se remueve.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("detrend_smoothness_priors expects a 1-D signal")
    n = x.size
    d2 = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n))
    a = sparse.identity(n) + lam**2 * (d2.T @ d2)
    trend = spsolve(a.tocsc(), x)
    return x - trend

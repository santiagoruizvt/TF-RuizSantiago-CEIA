"""Métodos rPPG clásicos: traza RGB de la piel -> señal de pulso.

Interfaz única (docs/arquitectura_pipeline_clasico.md §3.6):

    pulse = method(rgb, fs, **params)

- rgb: (N, 3) media espacial de R, G y B sobre la piel, muestreada a `fs` Hz
  sobre una grilla uniforme (ver `rppg.dsp.resample`), sin NaN.
- pulse: (N,) señal de pulso con media cero. Su escala es arbitraria: el
  protocolo común solo busca el pico espectral.

El filtrado pasa-banda y la estimación de HR no son parte del método: los
aplica el protocolo común (`rppg.eval.protocol`), igual que a PhysNet y a la
PPG de referencia (mitigación de R5).
"""

from collections.abc import Callable

import numpy as np

from rppg.dsp.filters import bandpass, detrend_linear, detrend_smoothness_priors

Method = Callable[..., np.ndarray]


def _check_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=float)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb must have shape (N, 3), got {rgb.shape}")
    if not np.isfinite(rgb).all():
        raise ValueError("rgb contains NaN or inf; resample over gaps first")
    if (rgb <= 0).any():
        raise ValueError("rgb must be positive (mean pixel intensities)")
    return rgb


def _window_length(window_s: float, fs: float, n: int) -> int:
    length = int(round(window_s * fs))
    if length < 2:
        raise ValueError(f"window_s={window_s} is too short for fs={fs}")
    if length > n:
        raise ValueError(f"window of {length} samples is longer than the signal ({n})")
    return length


def green(
    rgb: np.ndarray, fs: float, detrend: str = "linear", detrend_lambda: float = 100.0
) -> np.ndarray:
    """Canal verde normalizado por su media (Verkruysse et al., 2008).

    El verde es el canal con mayor modulación pulsátil, porque la hemoglobina
    absorbe más en esa banda. Es el baseline más simple: no compensa cambios de
    iluminación ni movimiento, que afectan a los tres canales por igual.

    detrend: "none", "linear" o "smoothness_priors" (Tarvainen et al., 2002,
    con `detrend_lambda`).
    """
    rgb = _check_rgb(rgb)
    g = rgb[:, 1] / rgb[:, 1].mean() - 1.0
    if detrend == "linear":
        return detrend_linear(g)
    if detrend == "smoothness_priors":
        return detrend_smoothness_priors(g, detrend_lambda)
    if detrend == "none":
        return g - g.mean()
    raise ValueError(f"Unknown detrend: {detrend!r}")


def _moving_mean(x: np.ndarray, length: int) -> np.ndarray:
    """Media móvil centrada por columna, con ventana truncada en los bordes."""
    kernel = np.ones(length)
    sums = np.column_stack([np.convolve(x[:, c], kernel, mode="same") for c in range(x.shape[1])])
    counts = np.convolve(np.ones(len(x)), kernel, mode="same")
    return sums / counts[:, None]


def chrom(
    rgb: np.ndarray,
    fs: float,
    window_s: float = 1.6,
    band_hz: tuple[float, float] = (0.7, 4.0),
    filter_order: int = 3,
) -> np.ndarray:
    """CHROM (de Haan & Jeanne, 2013): combinación de crominancias.

    1. Normaliza cada canal por su media local (ventana `window_s`): elimina el
       color de la piel y la intensidad de la luz, y deja las variaciones
       relativas.
    2. Arma dos señales de crominancia, X = 3R - 2G e Y = 1,5R + G - 1,5B. Con
       luz blanca, una variación de intensidad o especular afecta a R, G y B en
       la misma proporción y se anula en ambas; el pulso no.
    3. Filtra X e Y en la banda de HR y, en cada ventana, combina
       S = Xf - alpha * Yf con alpha = std(Xf) / std(Yf). Alpha se ajusta por
       ventana para cancelar la distorsión que quede.
    4. Une las ventanas (salto de media ventana) con pesos de Hann
       (*overlap-add*).

    Decisión de implementación: la normalización usa una media móvil sobre
    toda la señal y el pasa-banda se aplica a la señal completa, en lugar de
    dentro de cada ventana. Filtrar ventanas de ~1,6 s con un Butterworth de
    fase cero no es posible (son más cortas que el transitorio del filtro).
    """
    rgb = _check_rgb(rgb)
    n = len(rgb)
    length = _window_length(window_s, fs, n)
    norm = rgb / _moving_mean(rgb, length)
    r, g, b = norm[:, 0], norm[:, 1], norm[:, 2]
    xf = bandpass(3 * r - 2 * g, fs, *band_hz, order=filter_order)
    yf = bandpass(1.5 * r + g - 1.5 * b, fs, *band_hz, order=filter_order)

    hop = max(1, length // 2)
    starts = list(range(0, n - length + 1, hop))
    if starts[-1] + length < n:
        starts.append(n - length)
    w = np.hanning(length + 2)[1:-1]  # sin ceros: todo frame tiene peso
    out, weight = np.zeros(n), np.zeros(n)
    for s in starts:
        xw, yw = xf[s : s + length], yf[s : s + length]
        alpha = xw.std() / (yw.std() + 1e-12)
        sw = xw - alpha * yw
        out[s : s + length] += w * (sw - sw.mean())
        weight[s : s + length] += w
    return out / weight


def pos(rgb: np.ndarray, fs: float, window_s: float = 1.6) -> np.ndarray:
    """POS (Wang et al., 2017): proyección en el plano ortogonal a la piel.

    Implementa el algoritmo 1 del paper. Para cada ventana deslizante de
    `window_s` (paso de una muestra):
    1. normaliza cada canal por su media en la ventana (temporal normalization);
    2. proyecta sobre el plano ortogonal al tono de piel normalizado [1, 1, 1]
       con P = [[0, 1, -1], [-2, 1, 1]]. Una variación de intensidad común a los
       tres canales cae fuera de ese plano y se anula;
    3. combina las dos proyecciones, h = S1 + (std(S1) / std(S2)) * S2 (*alpha
       tuning*), para cancelar la distorsión especular que queda en el plano;
    4. suma h, con media cero, a la salida (*overlap-add*).
    """
    rgb = _check_rgb(rgb)
    n = len(rgb)
    length = _window_length(window_s, fs, n)
    projection = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    out = np.zeros(n)
    for m in range(n - length + 1):
        c = rgb[m : m + length]
        s = (c / c.mean(axis=0)) @ projection.T  # (L, 2)
        h = s[:, 0] + (s[:, 0].std() / (s[:, 1].std() + 1e-12)) * s[:, 1]
        out[m : m + length] += h - h.mean()
    return out


METHODS: dict[str, Method] = {"green": green, "chrom": chrom, "pos": pos}


def get_method(name: str) -> Method:
    try:
        return METHODS[name]
    except KeyError:
        raise ValueError(f"Unknown method {name!r}; choose from {sorted(METHODS)}") from None

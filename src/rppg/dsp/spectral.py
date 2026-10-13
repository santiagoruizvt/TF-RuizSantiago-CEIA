"""Estimación espectral de frecuencias (HR, RR) en ventanas deslizantes. Compartido con DL."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import periodogram, welch


@dataclass
class RateEstimate:
    t_center: np.ndarray  # (W,) segundos, centro de cada ventana
    rate_per_min: np.ndarray  # (W,) latidos o respiraciones por minuto; NaN si es inválida
    snr_db: np.ndarray  # (W,) SNR del pico; NaN si la ventana es inválida
    valid: np.ndarray  # (W,) bool


def power_spectrum(
    x: np.ndarray,
    fs: float,
    method: str = "periodogram",
    nfft: int | None = None,
    welch_segment_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PSD unilateral de una señal 1-D con ventana de Hann.

    `periodogram` usa la ventana completa (mejor resolución, 1/T Hz). `welch`
    promedia segmentos más cortos: menos varianza, resolución más gruesa.
    nfft solo agrega ceros: afina la ubicación del pico, no la resolución.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    nfft = max(nfft or 0, len(x))
    if method == "periodogram":
        return periodogram(x, fs=fs, window="hann", nfft=nfft, detrend=False)
    if method == "welch":
        nperseg = (
            len(x) if welch_segment_s is None else min(len(x), int(round(welch_segment_s * fs)))
        )
        return welch(
            x, fs=fs, window="hann", nperseg=nperseg, nfft=max(nfft, nperseg), detrend=False
        )
    raise ValueError(f"Unknown spectral method: {method}")


def dominant_frequency(
    x: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    method: str = "periodogram",
    nfft: int | None = None,
    welch_segment_s: float | None = None,
    snr_halfwidth_hz: float = 0.1,
) -> tuple[float, float]:
    """Frecuencia (Hz) del pico de la PSD dentro de band_hz y su SNR en dB.

    El pico se refina con interpolación parabólica sobre el log de la PSD. El
    SNR sigue a de Haan & Jeanne (2013): potencia a +-snr_halfwidth_hz del pico
    y de su primer armónico, sobre la potencia restante en la banda.
    """
    f, p = power_spectrum(x, fs, method=method, nfft=nfft, welch_segment_s=welch_segment_s)
    in_band = (f >= band_hz[0]) & (f <= band_hz[1])
    if not in_band.any():
        raise ValueError("No spectral bins inside the band; increase nfft or the window length")
    band_idx = np.flatnonzero(in_band)
    k = band_idx[np.argmax(p[band_idx])]

    f_peak = f[k]
    if 0 < k < len(f) - 1 and np.all(p[k - 1 : k + 2] > 0):
        a, b, c = np.log(p[k - 1 : k + 2])
        denom = a - 2 * b + c
        if denom != 0:
            f_peak = f[k] + 0.5 * (a - c) / denom * (f[1] - f[0])

    signal_mask = (np.abs(f - f_peak) <= snr_halfwidth_hz) | (
        np.abs(f - 2 * f_peak) <= snr_halfwidth_hz
    )
    p_signal = p[signal_mask].sum()
    p_noise = p[in_band & ~signal_mask].sum()
    snr_db = 10 * np.log10(p_signal / p_noise) if p_noise > 0 else np.inf
    return float(f_peak), float(snr_db)


def sliding_windows(n: int, fs: float, window_s: float, step_s: float) -> list[slice]:
    win, step = int(round(window_s * fs)), int(round(step_s * fs))
    if win > n:
        return []
    return [slice(s, s + win) for s in range(0, n - win + 1, step)]


def estimate_rate(
    x: np.ndarray,
    fs: float,
    band_hz: tuple[float, float],
    window_s: float,
    step_s: float,
    valid: np.ndarray | None = None,
    min_valid_fraction: float = 1.0,
    **spectral_kwargs,
) -> RateEstimate:
    """Frecuencia dominante (por minuto) en cada ventana deslizante de una señal uniforme.

    Una ventana es inválida si la fracción de muestras válidas es menor a
    min_valid_fraction. Las ventanas inválidas conservan su lugar con NaN, así
    las estimaciones de dos señales sobre la misma grilla quedan alineadas
    ventana a ventana.
    """
    x = np.asarray(x, dtype=float)
    valid = np.ones(len(x), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    windows = sliding_windows(len(x), fs, window_s, step_s)
    centers = np.array([(w.start + w.stop - 1) / 2 / fs for w in windows])
    rates = np.full(len(windows), np.nan)
    snrs = np.full(len(windows), np.nan)
    ok = np.zeros(len(windows), dtype=bool)
    for i, w in enumerate(windows):
        if valid[w].mean() < min_valid_fraction:
            continue
        f_peak, snr = dominant_frequency(x[w], fs, band_hz, **spectral_kwargs)
        rates[i], snrs[i], ok[i] = 60.0 * f_peak, snr, True
    return RateEstimate(t_center=centers, rate_per_min=rates, snr_db=snrs, valid=ok)

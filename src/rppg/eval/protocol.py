"""Protocolo de evaluación común: señal de pulso -> HR por ventana -> comparación.

Lo usan los dos enfoques con los mismos parámetros (`configs/eval.yaml`), así
las métricas son comparables (R5). La referencia es la PPG del oxímetro con el
mismo post-procesamiento; la HR que reporta el equipo no se usa.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from rppg.dsp.filters import bandpass
from rppg.dsp.resample import resample_uniform
from rppg.dsp.spectral import RateEstimate, estimate_rate
from rppg.eval.metrics import summarize


def hr_windows(t: np.ndarray, pulse: np.ndarray, cfg: Mapping[str, Any]) -> RateEstimate:
    """HR por ventana de una señal de pulso con timestamps (posiblemente irregulares)."""
    fs, hr = cfg["fs"], cfg["hr"]
    u = resample_uniform(t, pulse, fs, max_gap_s=cfg["max_gap_s"])
    y = bandpass(u.x, fs, *hr["band_hz"], order=hr["filter_order"])
    return estimate_rate(
        y,
        fs,
        tuple(hr["band_hz"]),
        hr["window_s"],
        hr["step_s"],
        valid=u.valid,
        min_valid_fraction=hr["min_valid_fraction"],
        method=hr["method"],
        nfft=hr["nfft"],
    )


def compare_hr(
    subject: str,
    t: np.ndarray,
    pulse: np.ndarray,
    t_ref: np.ndarray,
    ppg_ref: np.ndarray,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    """Tabla ventana a ventana: HR estimada vs. HR de referencia de un sujeto.

    Ambas señales se llevan a la misma grilla. Si la estimada es más corta que
    la referencia (p. ej. la red no cubre los últimos frames), solo se comparan
    las ventanas comunes.
    """
    est = hr_windows(t, pulse, cfg)
    ref = hr_windows(t_ref, ppg_ref, cfg)
    n = min(len(est.rate_per_min), len(ref.rate_per_min))
    return pd.DataFrame(
        {
            "subject": subject,
            "t_center": ref.t_center[:n],
            "hr_est": est.rate_per_min[:n],
            "hr_ref": ref.rate_per_min[:n],
            "snr_est_db": est.snr_db[:n],
            "valid": est.valid[:n] & ref.valid[:n],
        }
    )


def summarize_windows(windows: pd.DataFrame) -> pd.DataFrame:
    """Métricas globales (todas las ventanas válidas) y por sujeto."""
    valid = windows[windows["valid"]]
    rows = [{"subject": "ALL", **summarize(valid["hr_est"], valid["hr_ref"])}]
    for subject, group in valid.groupby("subject", sort=False):
        rows.append({"subject": subject, **summarize(group["hr_est"], group["hr_ref"])})
    return pd.DataFrame(rows).set_index("subject")

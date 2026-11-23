"""Tests del pipeline clásico: métodos rPPG, combinación de ROIs y alineación con la referencia."""

from pathlib import Path

import numpy as np
import pytest

from rppg.classical.methods import METHODS, chrom, get_method, green, pos
from rppg.classical.pipeline import (
    combine_rois,
    detection_validity,
    evaluate_subject,
    pulse_from_track,
)
from rppg.dsp.spectral import dominant_frequency
from rppg.eval.protocol import hr_windows
from rppg.face.extract import FaceTrack
from rppg.utils.config import load_yaml

ROOT = Path(__file__).resolve().parents[1]
EVAL_CFG = load_yaml(ROOT / "configs" / "eval.yaml")
FS = 30.0

# Variación relativa de R, G y B por el volumen de sangre (dirección del pulso):
# el verde es el canal más modulado. Valores ilustrativos para el test.
PULSE_DIRECTION = np.array([0.33, 0.77, 0.53])
SKIN = np.array([180.0, 130.0, 110.0])


def synthetic_rgb(
    seconds=30.0, hr_bpm=72.0, pulse_amp=0.004, flicker_bpm=None, flicker_amp=0.04, seed=0
):
    """Traza RGB de piel con pulso y, opcionalmente, un parpadeo de iluminación.

    El parpadeo multiplica los tres canales por igual (luz blanca que cambia de
    intensidad), que es lo que CHROM y POS cancelan y GREEN no.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    pulse = np.sin(2 * np.pi * hr_bpm / 60 * t)
    intensity = 1.0
    if flicker_bpm is not None:
        intensity = 1 + flicker_amp * np.sin(2 * np.pi * flicker_bpm / 60 * t + 0.3)
    rgb = SKIN * (1 + pulse_amp * pulse[:, None] * PULSE_DIRECTION)
    rgb = rgb * np.asarray(intensity)[..., None] if flicker_bpm is not None else rgb
    return t, rgb + rng.normal(0, 0.02, rgb.shape)


def peak_bpm(x):
    return 60 * dominant_frequency(x, FS, (0.7, 4.0), nfft=8192)[0]


# --- métodos -----------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(METHODS))
def test_methods_recover_pulse_rate(name):
    _, rgb = synthetic_rgb(hr_bpm=72)
    out = get_method(name)(rgb, FS)
    assert out.shape == (len(rgb),) and np.isfinite(out).all()
    assert peak_bpm(out) == pytest.approx(72, abs=1)


def test_chrom_and_pos_cancel_common_intensity_changes_but_green_does_not():
    """Un parpadeo de luz 10 veces más fuerte que el pulso, igual en los tres canales."""
    _, rgb = synthetic_rgb(hr_bpm=72, flicker_bpm=114)
    assert peak_bpm(green(rgb, FS)) == pytest.approx(114, abs=1)
    assert peak_bpm(chrom(rgb, FS)) == pytest.approx(72, abs=1)
    assert peak_bpm(pos(rgb, FS)) == pytest.approx(72, abs=1)


def test_pos_projection_removes_exactly_a_pure_intensity_change():
    t = np.arange(300) / FS
    rgb = SKIN * (1 + 0.05 * np.sin(2 * np.pi * 1.5 * t))[:, None]
    np.testing.assert_allclose(pos(rgb, FS), 0, atol=1e-9)


def test_green_detrend_options():
    t, rgb = synthetic_rgb()
    rgb = rgb * (1 + 0.002 * t)[:, None]  # deriva lenta de iluminación
    for detrend in ("none", "linear", "smoothness_priors"):
        assert peak_bpm(green(rgb, FS, detrend=detrend)) == pytest.approx(72, abs=1)
    with pytest.raises(ValueError):
        green(rgb, FS, detrend="wavelet")


def test_methods_validate_input():
    _, rgb = synthetic_rgb(seconds=5)
    with pytest.raises(ValueError):
        pos(rgb[:, :2], FS)
    bad = rgb.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        chrom(bad, FS)
    with pytest.raises(ValueError):
        pos(rgb[:10], FS, window_s=1.6)  # ventana más larga que la señal
    with pytest.raises(ValueError):
        get_method("ica")


# --- combinación de ROIs y validez --------------------------------------------


def test_combine_rois_weights_by_pixels_and_keeps_missing_frames():
    rgb = np.array(
        [
            [[10.0, 10.0, 10.0], [40.0, 40.0, 40.0]],
            [[np.nan] * 3, [np.nan] * 3],
            [[np.nan] * 3, [20.0, 20.0, 20.0]],
        ]
    )
    n_pixels = np.array([[100, 300], [0, 0], [0, 50]])
    out = combine_rois(rgb, n_pixels)
    np.testing.assert_allclose(out[0], 32.5)  # (10*100 + 40*300) / 400
    assert np.isnan(out[1]).all()
    np.testing.assert_allclose(out[2], 20.0)
    np.testing.assert_allclose(combine_rois(rgb, n_pixels, rois=[0])[0], 10.0)


def test_detection_validity_marks_long_gaps_and_edges():
    t = np.arange(100) / 10.0  # 10 fps
    detected = np.ones(100, dtype=bool)
    detected[:5] = False  # sin cara los primeros 0,5 s
    detected[40:43] = False  # hueco corto: 0,4 s entre detecciones
    detected[60:70] = False  # hueco largo: 1,1 s
    grid = np.arange(0, 9.9, 1 / 30)
    valid = detection_validity(t, detected, grid, max_gap_s=0.5)
    assert not valid[grid < 0.5 - 1e-9].any()
    assert valid[(grid > 3.9) & (grid < 4.3)].all()
    assert not valid[(grid > 5.95) & (grid < 6.95)].any()
    assert valid[(grid > 7.1) & (grid < 9.8)].all()


def test_hr_windows_respects_external_validity():
    t = np.arange(0, 40, 1 / FS)
    pulse = np.sin(2 * np.pi * 1.2 * t)
    valid = t >= 15
    est = hr_windows(t, pulse, EVAL_CFG, valid=valid)
    window = EVAL_CFG["hr"]["window_s"]
    starts = est.t_center - (window - 1 / FS) / 2
    np.testing.assert_array_equal(est.valid, starts >= 15 - 1e-9)


def make_track(t, rgb, detected):
    rgb_roi = np.where(detected[:, None], rgb, np.nan)[:, None, :]
    return FaceTrack(
        t=t,
        detected=detected,
        rgb=rgb_roi,
        n_pixels=np.where(detected, 500, 0)[:, None],
        roi_names=["face"],
        crops=None,
    )


def test_pulse_grid_starts_at_first_timestamp_even_without_face():
    t, rgb = synthetic_rgb(seconds=30)
    detected = np.ones(len(t), dtype=bool)
    detected[: int(3 * FS)] = False  # sin cara los primeros 3 s
    grid, pulse, valid = pulse_from_track(make_track(t, rgb, detected), "pos", FS, 0.5)
    assert grid[0] == pytest.approx(t[0])
    assert np.isfinite(pulse).all()
    assert not valid[grid < 3 - 1 / FS].any() and valid[grid > 3.1].all()


def test_evaluate_subject_aligns_with_reference_and_invalidates_missing_face():
    t, rgb = synthetic_rgb(seconds=40, hr_bpm=72)
    detected = np.ones(len(t), dtype=bool)
    detected[: int(3 * FS)] = False
    ppg = np.sin(2 * np.pi * 1.2 * t)  # referencia: 72 bpm desde el primer instante
    windows = evaluate_subject(
        "s1", make_track(t, rgb, detected), t, ppg, "pos", EVAL_CFG, params={"window_s": 1.6}
    )
    assert (windows["method"] == "pos").all()
    start = windows["t_center"] - (EVAL_CFG["hr"]["window_s"] - 1 / FS) / 2
    first_valid = start >= 3 - 1e-9
    assert not windows.loc[~first_valid, "valid"].any()
    ok = windows[windows["valid"]]
    assert len(ok) > 0
    np.testing.assert_allclose(ok["hr_est"], ok["hr_ref"], atol=1.0)

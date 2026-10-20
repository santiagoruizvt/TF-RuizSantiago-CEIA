import numpy as np
import pytest

from rppg.dsp.filters import bandpass, detrend_smoothness_priors
from rppg.dsp.resample import resample_uniform
from rppg.dsp.spectral import dominant_frequency, estimate_rate, sliding_windows

FS = 30.0
HR_BAND = (0.7, 4.0)
RR_BAND = (0.1, 0.5)


def tone(f_hz, duration_s, fs=FS, phase=0.0):
    t = np.arange(int(duration_s * fs)) / fs
    return t, np.sin(2 * np.pi * f_hz * t + phase)


# --- filtros ----------------------------------------------------------------


def test_bandpass_keeps_in_band_and_rejects_out_of_band():
    _, pulse = tone(1.2, 30)
    _, drift = tone(0.05, 30)
    _, flicker = tone(10.0, 30)
    y = bandpass(pulse + 2 * drift + flicker, FS, *HR_BAND)
    core = slice(150, -150)  # descarta los transitorios de borde de filtfilt
    assert np.corrcoef(y[core], pulse[core])[0, 1] > 0.99
    assert np.std(y[core]) == pytest.approx(np.std(pulse[core]), rel=0.05)


def test_bandpass_is_zero_phase():
    _, pulse = tone(1.2, 30)
    y = bandpass(pulse, FS, *HR_BAND)
    lag = np.argmax(np.correlate(y[150:-150], pulse[150:-150], mode="full")) - (len(pulse) - 301)
    assert lag == 0


def test_bandpass_rejects_band_above_nyquist():
    with pytest.raises(ValueError):
        bandpass(np.zeros(100), FS, 0.7, 20.0)


def test_smoothness_priors_removes_slow_trend():
    t, pulse = tone(1.2, 30)
    trend = 3 * t / t[-1] + np.sin(2 * np.pi * 0.02 * t)
    y = detrend_smoothness_priors(pulse + trend, lam=100)
    assert np.corrcoef(y, pulse)[0, 1] > 0.98


# --- espectral --------------------------------------------------------------


@pytest.mark.parametrize("bpm", [48, 72, 110, 180])
def test_dominant_frequency_recovers_hr_with_noise(bpm):
    rng = np.random.default_rng(0)
    _, x = tone(bpm / 60, 10)
    f, snr = dominant_frequency(x + 0.5 * rng.standard_normal(len(x)), FS, HR_BAND, nfft=4096)
    assert 60 * f == pytest.approx(bpm, abs=1.0)
    assert snr > 0


def test_parabolic_interpolation_beats_bin_resolution():
    # Ventana de 10 s -> bins cada 6 bpm; 75 bpm cae entre dos bins.
    _, x = tone(75 / 60, 10)
    f, _ = dominant_frequency(x, FS, HR_BAND)
    assert abs(60 * f - 75) < 1.5


def test_dominant_frequency_ignores_out_of_band_peak():
    _, pulse = tone(1.0, 10)
    _, resp = tone(0.25, 10)
    f, _ = dominant_frequency(pulse + 5 * resp, FS, HR_BAND, nfft=4096)
    assert 60 * f == pytest.approx(60, abs=1.0)
    f, _ = dominant_frequency(pulse + 5 * resp, FS, RR_BAND, nfft=4096)
    assert 60 * f == pytest.approx(15, abs=0.5)


def test_welch_method_runs():
    _, x = tone(1.5, 20)
    f, _ = dominant_frequency(x, FS, HR_BAND, method="welch", welch_segment_s=10, nfft=4096)
    assert 60 * f == pytest.approx(90, abs=1.0)


def test_sliding_windows_count_and_length():
    w = sliding_windows(n=int(60 * FS), fs=FS, window_s=10, step_s=1)
    assert len(w) == 51
    assert all(s.stop - s.start == 300 for s in w)
    assert sliding_windows(n=100, fs=FS, window_s=10, step_s=1) == []


def test_estimate_rate_tracks_changing_hr_and_flags_invalid_windows():
    _, a = tone(1.0, 30)
    _, b = tone(1.5, 30)
    x = np.concatenate([a, b])
    valid = np.ones(len(x), dtype=bool)
    valid[: int(2 * FS)] = False
    est = estimate_rate(x, FS, HR_BAND, window_s=10, step_s=5, valid=valid, nfft=4096)
    assert not est.valid[0] and np.isnan(est.rate_per_min[0])
    assert est.rate_per_min[1] == pytest.approx(60, abs=1.5)
    assert est.rate_per_min[-1] == pytest.approx(90, abs=1.5)
    assert len(est.t_center) == len(est.rate_per_min)


# --- remuestreo -------------------------------------------------------------


def test_resample_irregular_timestamps_preserves_frequency():
    rng = np.random.default_rng(1)
    t = np.cumsum(rng.uniform(0.02, 0.045, size=2000))
    x = np.sin(2 * np.pi * 1.2 * t)
    u = resample_uniform(t, x, fs=FS)
    assert np.allclose(np.diff(u.t), 1 / FS)
    assert np.max(np.abs(u.x - np.sin(2 * np.pi * 1.2 * u.t))) < 0.05
    f, _ = dominant_frequency(u.x, FS, HR_BAND, nfft=8192)
    assert 60 * f == pytest.approx(72, abs=0.5)


def test_resample_flags_gaps_and_handles_nan_and_duplicates():
    t = np.array([0.0, 0.1, 0.1, 0.2, 1.5, 1.6, 1.7])
    x = np.array([[0, 0], [1, 1], [3, 3], [2, 2], [5, 5], [np.nan, 0], [7, 7]], dtype=float)
    u = resample_uniform(t, x, fs=10.0, max_gap_s=0.5)
    assert u.x.shape == (18, 2)
    assert u.x[1, 0] == pytest.approx(2.0)  # timestamps duplicados promediados
    gap = (u.t > 0.2 + 1e-9) & (u.t < 1.5 - 1e-9)
    assert not u.valid[gap].any()
    assert u.valid[~gap].all()  # 1,5 -> 1,7 abarca 0,2 s al descartar la fila con NaN

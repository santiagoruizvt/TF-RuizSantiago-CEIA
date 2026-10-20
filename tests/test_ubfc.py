"""Tests del loader de UBFC-rPPG y de la cadena frames -> HR.

Los mismos chequeos de formato corren sobre el dataset sintético (siempre) y
sobre la copia local real (si está). Los de exactitud solo usan el sintético,
porque ahí la HR verdadera se conoce exactamente.
"""

import numpy as np
import pytest

from rppg.data.ubfc import UBFCrPPG
from rppg.dsp.filters import bandpass
from rppg.dsp.resample import resample_uniform
from rppg.dsp.spectral import estimate_rate
from rppg.eval.metrics import mae

FS = 30.0
HR_BAND = (0.7, 4.0)


@pytest.fixture(params=["synthetic", "real"])
def ds(request) -> UBFCrPPG:
    if request.param == "synthetic":
        return UBFCrPPG(request.getfixturevalue("synthetic_ubfc").root)
    return UBFCrPPG(request.getfixturevalue("real_ubfc_root"))


def window_hr(t, x, valid_gap_s=0.5):
    u = resample_uniform(t, x, FS, max_gap_s=valid_gap_s)
    y = bandpass(u.x, FS, *HR_BAND)
    return estimate_rate(y, FS, HR_BAND, window_s=10, step_s=1, valid=u.valid, nfft=4096)


# --- formato (sintético y real) ---------------------------------------------


def test_subjects_sorted_numerically(ds):
    nums = [int(s.removeprefix("subject")) for s in ds.subjects()]
    assert nums and nums == sorted(nums)


def test_ground_truth_has_one_sample_per_frame(ds):
    subject = ds.subjects()[0]
    gt = ds.ground_truth(subject)
    assert len(gt.ppg) == len(gt.hr_device) == len(gt.t) == ds.frame_count(subject)
    assert np.all(np.diff(gt.t) >= 0)


def test_frames_are_rgb_uint8(ds):
    frame = next(ds.frames(ds.subjects()[0]))
    assert frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype == np.uint8


def test_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        UBFCrPPG(tmp_path / "nope")


# --- exactitud (solo sintético) ---------------------------------------------


def test_incomplete_subject_folders_are_ignored(synthetic_ubfc):
    assert UBFCrPPG(synthetic_ubfc.root).subjects() == list(synthetic_ubfc.true_hr_bpm)


def test_frames_are_returned_as_rgb_not_bgr(synthetic_ubfc):
    ds = UBFCrPPG(synthetic_ubfc.root)
    frame = next(ds.frames(ds.subjects()[0]))
    h, w, _ = frame.shape
    r, g, b = frame[h // 2, w // 2].astype(int)
    assert r > g > b  # el parche se escribió en BGR como (110, ~150, 170)


def test_reference_hr_from_ppg_matches_truth(synthetic_ubfc):
    ds = UBFCrPPG(synthetic_ubfc.root)
    for subject, true_hr in synthetic_ubfc.true_hr_bpm.items():
        gt = ds.ground_truth(subject)
        est = window_hr(gt.t, gt.ppg)
        assert est.valid.any() and not est.valid.all()  # el hueco de 1 s invalida ventanas
        assert np.nanmax(np.abs(est.rate_per_min - true_hr)) < 2.0


def test_green_trace_from_frames_recovers_hr(synthetic_ubfc):
    """Cadena completa: frames -> media del verde -> remuestreo -> filtro -> HR."""
    ds = UBFCrPPG(synthetic_ubfc.root)
    for subject, true_hr in synthetic_ubfc.true_hr_bpm.items():
        gt = ds.ground_truth(subject)
        green = np.array([f[..., 1].mean() for f in ds.frames(subject)])
        est = window_hr(gt.t, green)
        ref = window_hr(gt.t, gt.ppg)
        assert np.nanmax(np.abs(est.rate_per_min - true_hr)) < 2.0
        assert mae(est.rate_per_min, ref.rate_per_min) < 1.0


def test_ignoring_timestamps_biases_hr(synthetic_ubfc):
    """Justifica remuestrear con timestamps: asumir fps fijo sesga la HR estimada."""
    ds = UBFCrPPG(synthetic_ubfc.root)
    subject, true_hr = next(iter(synthetic_ubfc.true_hr_bpm.items()))
    gt = ds.ground_truth(subject)
    nominal_t = np.arange(len(gt.t)) / FS
    with_ts = window_hr(gt.t, gt.ppg)
    without_ts = window_hr(nominal_t, gt.ppg)
    err_with = np.nanmedian(np.abs(with_ts.rate_per_min - true_hr))
    err_without = np.nanmedian(np.abs(without_ts.rate_per_min - true_hr))
    assert err_with < 1.0
    assert err_without > err_with

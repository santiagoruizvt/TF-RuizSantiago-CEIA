"""Tests del front-end facial: ROIs, recortes, suavizado, extracción y detector."""

from itertools import islice
from pathlib import Path

import numpy as np
import pytest

from rppg.data.ubfc import UBFCrPPG
from rppg.dsp.filters import bandpass
from rppg.dsp.resample import resample_uniform
from rppg.dsp.spectral import estimate_rate
from rppg.face.extract import cached_hash, extract_video, load_traces, save_track
from rppg.face.landmarker import LandmarkSmoother, MonotonicClock
from rppg.face.roi import face_crop, polygon_mask, roi_means, skin_mask
from rppg.utils.config import config_hash, load_yaml

ROOT = Path(__file__).resolve().parents[1]
FACE_CFG = load_yaml(ROOT / "configs" / "face.yaml")


def square(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def fake_landmarks(*polygons):
    """Landmarks concatenando polígonos; devuelve también los índices de cada uno."""
    pts, rois, start = [], {}, 0
    for k, poly in enumerate(polygons):
        pts.append(poly)
        rois[f"roi{k}"] = list(range(start, start + len(poly)))
        start += len(poly)
    return np.vstack(pts), rois


# --- ROIs y recortes ---------------------------------------------------------


def test_polygon_mask_area():
    mask = polygon_mask((20, 30), square(5, 2, 14, 11))
    assert mask.sum() == 10 * 10
    assert mask[2, 5] and mask[11, 14] and not mask[12, 14]


def test_roi_means_exact_colors_and_counts():
    frame = np.zeros((60, 80, 3), dtype=np.uint8)
    frame[10:20, 10:30] = (200, 100, 50)
    frame[30:50, 40:60] = (10, 20, 30)
    landmarks, rois = fake_landmarks(square(10, 10, 29, 19), square(40, 30, 59, 49))
    means, counts = roi_means(frame, landmarks, rois)
    np.testing.assert_allclose(means, [[200, 100, 50], [10, 20, 30]])
    np.testing.assert_array_equal(counts, [10 * 20, 20 * 20])


def test_roi_outside_frame_is_nan_and_partial_roi_is_clipped():
    frame = np.full((40, 40, 3), 7, dtype=np.uint8)
    landmarks, rois = fake_landmarks(square(100, 100, 120, 120), square(30, 30, 49, 49))
    means, counts = roi_means(frame, landmarks, rois)
    assert np.isnan(means[0]).all() and counts[0] == 0
    np.testing.assert_allclose(means[1], [7, 7, 7])
    assert counts[1] == 10 * 10


def test_skin_mask_excludes_non_skin_pixels():
    frame = np.zeros((10, 20, 3), dtype=np.uint8)
    frame[:, :10] = (200, 140, 120)  # tono de piel
    frame[:, 10:] = (20, 160, 60)  # verde: fuera del rango de piel
    skin_cfg = FACE_CFG["skin_mask"]
    skin = skin_mask(frame, skin_cfg["cr_range"], skin_cfg["cb_range"])
    assert skin[:, :10].all() and not skin[:, 10:].any()
    landmarks, rois = fake_landmarks(square(0, 0, 19, 9))
    means, counts = roi_means(frame, landmarks, rois, skin)
    assert counts[0] == 100
    np.testing.assert_allclose(means[0], [200, 140, 120])


def test_face_crop_shape_and_border_padding():
    frame = np.random.default_rng(0).integers(0, 255, (100, 120, 3), dtype=np.uint8)
    center = face_crop(frame, square(40, 30, 80, 70), size=32, scale=1.5)
    corner = face_crop(frame, square(0, 0, 30, 30), size=32, scale=2.0)
    assert center.shape == corner.shape == (32, 32, 3)
    assert center.dtype == np.uint8


# --- suavizado y reloj -------------------------------------------------------


def test_monotonic_clock_handles_repeated_and_backward_timestamps():
    clock = MonotonicClock()
    out = [clock(t) for t in [0.0, 0.033, 0.033, 0.030, 0.1]]
    assert out == [0, 33, 34, 35, 100]


def test_landmark_smoother_ema_and_reset_on_gap():
    s = LandmarkSmoother(alpha=0.5)
    a, b = np.zeros((2, 2)), np.full((2, 2), 4.0)
    np.testing.assert_allclose(s(a), 0.0)
    np.testing.assert_allclose(s(b), 2.0)
    assert s(None) is None
    np.testing.assert_allclose(s(b), 4.0)  # tras un hueco no arrastra el estado viejo


def test_landmark_smoother_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        LandmarkSmoother(0.0)


# --- extracción --------------------------------------------------------------


def test_extract_video_with_fake_detector_recovers_hr(synthetic_ubfc, tmp_path):
    """Extracción completa sobre el dataset sintético con un detector falso.

    El detector devuelve un polígono dentro del parche que late; además "pierde"
    la cara en algunos frames para probar el manejo de huecos.
    """
    ds = UBFCrPPG(synthetic_ubfc.root)
    subject, true_hr = next(iter(synthetic_ubfc.true_hr_bpm.items()))
    gt = ds.ground_truth(subject)
    w, h = 64, 48
    patch = square(w // 4 + 2, h // 4 + 2, 3 * w // 4 - 3, 3 * h // 4 - 3)
    landmarks, rois = fake_landmarks(patch)
    lost = set(range(100, 110))

    calls = iter(range(len(gt.t)))

    def detector(frame, t):
        return None if next(calls) in lost else landmarks

    track = extract_video(ds.frames(subject), gt.t, detector, rois, crop_size=16)
    assert track.detection_rate == pytest.approx(1 - len(lost) / len(gt.t))
    assert np.isnan(track.rgb[100]).all() and not track.detected[100]
    assert track.crops.shape == (len(gt.t), 16, 16, 3)
    np.testing.assert_array_equal(track.crops[105], track.crops[99])  # repite el último

    save_track(track, tmp_path, "abc")
    assert cached_hash(tmp_path) == "abc"
    loaded = load_traces(tmp_path)
    np.testing.assert_array_equal(loaded.detected, track.detected)
    assert loaded.roi_names == ["roi0"]

    green = loaded.rgb[:, 0, 1]
    u = resample_uniform(loaded.t, green, 30.0)
    y = bandpass(u.x, 30.0, 0.7, 4.0)
    est = estimate_rate(y, 30.0, (0.7, 4.0), 10, 1, valid=u.valid, nfft=4096)
    assert np.nanmax(np.abs(est.rate_per_min - true_hr)) < 2.0


def test_extract_video_rejects_frame_timestamp_mismatch():
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)] * 3
    with pytest.raises(ValueError):
        extract_video(frames, [0.0, 0.1], lambda f, t: None, {"r": [0, 1, 2]})


def test_config_hash_is_order_independent():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


# --- detector real (requiere el modelo de MediaPipe) -------------------------


@pytest.fixture(scope="session")
def model_path():
    from rppg.face.landmarker import ensure_model

    m = FACE_CFG["model"]
    try:
        return ensure_model(ROOT / m["path"], m["url"], m["sha256"])
    except OSError as e:  # sin red
        pytest.skip(f"MediaPipe model unavailable: {e}")


def test_detector_finds_face_and_rois_land_on_skin(model_path):
    skimage_data = pytest.importorskip("skimage.data")
    from rppg.face.landmarker import N_LANDMARKS, VideoLandmarker

    image = skimage_data.astronaut()  # foto de dominio público (NASA)
    with VideoLandmarker(model_path) as detector:
        landmarks = detector(image, 0.0)
    assert landmarks is not None and landmarks.shape == (N_LANDMARKS, 2)
    means, counts = roi_means(image, landmarks, FACE_CFG["rois"])
    assert (counts > 50).all()
    # Piel: rojo > verde > azul en las tres ROIs.
    assert (means[:, 0] > means[:, 1]).all() and (means[:, 1] > means[:, 2]).all()


def test_detector_returns_none_without_face(model_path):
    from rppg.face.landmarker import VideoLandmarker

    with VideoLandmarker(model_path) as detector:
        assert detector(np.full((240, 320, 3), 128, dtype=np.uint8), 0.0) is None


def test_detector_on_real_ubfc_frames(model_path, real_ubfc_root):
    from rppg.face.landmarker import VideoLandmarker

    ds = UBFCrPPG(real_ubfc_root)
    subject = ds.subjects()[0]
    gt = ds.ground_truth(subject)
    with VideoLandmarker(model_path, smoothing=0.5) as detector:
        track = extract_video(
            islice(ds.frames(subject), 60),
            gt.t[:60],
            detector,
            FACE_CFG["rois"],
        )
    assert track.detection_rate >= 0.95  # requerimiento del plan

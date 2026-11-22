"""Tests del modelo de DL: arquitectura, pérdida, datos, inferencia y entrenamiento."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rppg.data.ubfc import UBFCrPPG  # noqa: E402
from rppg.dl.dataset import (  # noqa: E402
    Augmenter,
    ChunkDataset,
    SubjectClip,
    interpolate_frames,
    load_subject,
    standardize_frames,
)
from rppg.dl.infer import chunk_starts, overlap_weights, predict_signal  # noqa: E402
from rppg.dl.losses import neg_pearson  # noqa: E402
from rppg.dl.physnet import PhysNet  # noqa: E402
from rppg.dl.train import (  # noqa: E402
    _epochs_since_best,
    inference_options,
    load_model,
    signal_losses,
    train,
)
from rppg.face.extract import extract_video, save_track  # noqa: E402
from rppg.utils.config import load_yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVAL_CFG = load_yaml(ROOT / "configs" / "eval.yaml")


# --- arquitectura y pérdida ------------------------------------------------


def test_physnet_output_shape_and_frame_check():
    model = PhysNet(frames=32).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 32, 32, 32))
    assert out.shape == (2, 32)
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 16, 32, 32))


def test_physnet_parameter_count_matches_original():
    # Cantidad de parámetros de PhysNet_padding_Encoder_Decoder_MAX del
    # repositorio original, calculada con el código del autor (25/10/2026).
    # Con los mismos pesos, ambas implementaciones dan salidas idénticas.
    assert sum(p.numel() for p in PhysNet().parameters()) == 768_577


def test_neg_pearson_values():
    t = torch.linspace(0, 6.28, 100)
    y = torch.stack([torch.sin(t), torch.cos(t)])
    assert neg_pearson(y, y).item() == pytest.approx(0.0, abs=1e-5)
    assert neg_pearson(-y, y).item() == pytest.approx(2.0, abs=1e-5)
    assert neg_pearson(3 * y + 7, y).item() == pytest.approx(0.0, abs=1e-5)  # escala y offset


# --- datos -------------------------------------------------------------------


def make_clip(n=200, size=16, f_hz=1.2, fs=30.0, seed=0):
    """Video sintético cuyo verde late a f_hz, con su PPG."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    pulse = np.sin(2 * np.pi * f_hz * t)
    frames = np.full((n, size, size, 3), (170, 120, 100), dtype=np.float32)
    frames[..., 1] += 4 * pulse[:, None, None]
    frames += rng.normal(0, 1, frames.shape)
    return SubjectClip(
        "subjectX",
        np.clip(frames, 0, 255).astype(np.uint8),
        pulse.astype(np.float32),
        t,
        np.ones(n, dtype=bool),
    )


def test_standardize_frames_layout_and_stats():
    x = standardize_frames(make_clip(n=40).frames)
    assert x.shape == (3, 40, 16, 16) and x.dtype == np.float32
    np.testing.assert_allclose(x.mean(axis=(1, 2, 3)), 0, atol=1e-4)
    np.testing.assert_allclose(x.std(axis=(1, 2, 3)), 1, atol=1e-3)


def test_chunk_dataset_skips_chunks_without_detection():
    clip = make_clip(n=200)
    clip.detected[70] = False
    ds = ChunkDataset([clip], chunk_length=32, stride=32)
    starts = [s for _, s in ds.index]
    assert starts == [0, 32, 96, 128, 160]  # el tramo 64-95 contiene el frame 70
    x, y = ds[0]
    assert x.shape == (3, 32, 16, 16) and y.shape == (32,)
    assert float(y.mean()) == pytest.approx(0, abs=1e-5)


def test_augmenter_keeps_shape_and_range():
    frames = make_clip(n=32).frames
    aug = Augmenter({"flip_p": 1.0, "gain_drift": 0.1, "noise_std": 0.01}, seed=0)
    out = aug(frames)
    assert out.shape == frames.shape
    assert out.min() >= 0 and out.max() <= 255
    np.testing.assert_allclose(aug(frames[:, :, ::-1])[0].mean(), out[0].mean(), rtol=0.2)


def test_load_subject_from_extracted_crops(synthetic_ubfc, tmp_path):
    ds = UBFCrPPG(synthetic_ubfc.root)
    subject = ds.subjects()[0]
    gt = ds.ground_truth(subject)
    box = np.array([[8.0, 6.0], [56.0, 6.0], [56.0, 42.0], [8.0, 42.0]])
    track = extract_video(
        ds.frames(subject), gt.t, lambda f, t: box, {"r": [0, 1, 2, 3]}, crop_size=32
    )
    save_track(track, tmp_path / subject, "h")
    clip = load_subject(subject, tmp_path, synthetic_ubfc.root, input_size=16)
    assert clip.frames.shape == (len(gt.t), 16, 16, 3)
    np.testing.assert_allclose(clip.ppg, gt.ppg.astype(np.float32))


# --- inferencia --------------------------------------------------------------


class GreenProbe(torch.nn.Module):
    """'Modelo' que devuelve el verde medio de cada frame: permite verificar el armado."""

    frames = 32

    def forward(self, x):
        return x[:, 1].mean(dim=(2, 3))


@pytest.mark.parametrize("n", [96, 100, 131])
@pytest.mark.parametrize("stride,flip", [(None, False), (16, False), (8, True), (32, True)])
def test_predict_signal_covers_every_frame(n, stride, flip):
    clip = make_clip(n=n)
    out = predict_signal(
        GreenProbe(), clip.frames, chunk_length=32, stride=stride, flip_average=flip
    )
    assert out.shape == (n,) and np.isfinite(out).all()
    assert np.corrcoef(out, clip.ppg)[0, 1] > 0.9


@pytest.mark.parametrize("n,stride", [(96, 32), (100, 32), (100, 16), (131, 30)])
def test_chunk_starts_cover_video_and_end_aligned(n, stride):
    starts = chunk_starts(n, 32, stride)
    assert starts[0] == 0 and starts[-1] == n - 32
    assert all(0 < b - a <= stride for a, b in zip(starts, starts[1:], strict=False))
    with pytest.raises(ValueError):
        chunk_starts(n, 32, 0)
    with pytest.raises(ValueError):
        chunk_starts(n, 32, 33)


def test_overlap_weights_positive_and_symmetric():
    w = overlap_weights(32)
    assert w.shape == (32,) and (w > 0).all()
    np.testing.assert_allclose(w, w[::-1])


class LeftHalfProbe(torch.nn.Module):
    """Verde medio de la mitad izquierda: distingue el video del espejado."""

    frames = 32

    def forward(self, x):
        return x[:, 1, :, :, : x.shape[-1] // 2].mean(dim=(2, 3))


def test_predict_signal_flip_average_sees_the_mirrored_half():
    # El pulso está solo en la mitad derecha: la mitad izquierda no lo ve,
    # pero la versión espejada sí.
    clip = make_clip(n=96)
    frames = clip.frames.copy()
    frames[:, :, :8] = make_clip(n=96, f_hz=0.0, seed=1).frames[:, :, :8]
    plain = predict_signal(LeftHalfProbe(), frames, chunk_length=32)
    averaged = predict_signal(LeftHalfProbe(), frames, chunk_length=32, flip_average=True)
    assert abs(np.corrcoef(plain, clip.ppg)[0, 1]) < 0.5
    assert np.corrcoef(averaged, clip.ppg)[0, 1] > 0.6


def test_predict_signal_rejects_short_video():
    with pytest.raises(ValueError):
        predict_signal(GreenProbe(), make_clip(n=20).frames, chunk_length=32)


# --- entrenamiento -----------------------------------------------------------


def tiny_cfg(tmp_path, epochs):
    return {
        "seed": 0,
        "run_name": "tiny",
        "data": {"chunk_length": 32, "train_stride": 16, "input_size": 16},
        "augment": {
            "flip_p": 0.5,
            "gain_drift": 0.05,
            "noise_std": 0.005,
            "speed_p": 0.5,
            "speed_range": [0.8, 1.2],
        },
        "train": {
            "epochs": epochs,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "num_workers": 0,
            "amp": False,
            "device": "cpu",
            "select_by": "val_loss",
            "early_stopping_patience": None,
        },
        "infer": {"stride": 16, "flip_average": False},
    }


def test_inference_options_defaults_for_old_configs():
    old = {"data": {"chunk_length": 32}}  # checkpoints anteriores a la v0.1.7
    assert inference_options(old) == {"stride": 32, "flip_average": False}
    assert inference_options({**old, "infer": {"stride": None}})["stride"] == 32
    new = {**old, "infer": {"stride": 16, "flip_average": True}}
    assert inference_options(new) == {"stride": 16, "flip_average": True}


def test_training_learns_and_resumes(tmp_path):
    """Sobre pulsos sintéticos la pérdida baja; reanudar continúa el historial."""
    train_clips = [make_clip(n=400, f_hz=f, seed=i) for i, f in enumerate([1.0, 1.3, 1.6])]
    for i, c in enumerate(train_clips):
        c.subject = f"train{i}"
    val_clips = [make_clip(n=400, f_hz=1.2, seed=9)]
    run = tmp_path / "run"

    history = train(
        tiny_cfg(tmp_path, 4), EVAL_CFG, train_clips, val_clips, run, log=lambda m: None
    )
    assert (run / "last.pt").is_file() and (run / "best.pt").is_file()
    assert history["train_loss"].iloc[-1] < history["train_loss"].iloc[0]

    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    assert len(info["sessions"]) == 1 and info["sessions"][0]["start_epoch"] == 1
    assert history["commit"].notna().all()

    resumed = train(
        tiny_cfg(tmp_path, 5),
        EVAL_CFG,
        train_clips,
        val_clips,
        run,
        resume=True,
        log=lambda m: None,
    )
    assert list(resumed["epoch"]) == [1, 2, 3, 4, 5]
    np.testing.assert_allclose(resumed["train_loss"][:4], history["train_loss"])

    info = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    assert [s["start_epoch"] for s in info["sessions"]] == [1, 5]
    assert info["sessions"][1]["resumed"] is True

    assert history["val_loss"].between(0, 2).all()

    model, cfg, meta = load_model(run / "best.pt")
    assert meta["selected_by"] == "val_loss"
    assert cfg["data"]["chunk_length"] == 32 and model.frames == 32
    assert meta["provenance"]["git"]["commit"] is not None  # el repo de tests es git


# --- procedencia -------------------------------------------------------------


def test_git_info_outside_a_repository_is_empty(tmp_path):
    from rppg.utils.provenance import git_info

    info = git_info(tmp_path)
    assert info["commit"] is None and info["dirty"] is None


def test_git_info_inside_the_repository():
    from rppg.utils.provenance import git_info, provenance

    info = git_info()
    assert len(info["commit"]) == 40 and info["short"] == info["commit"][:7]
    assert isinstance(info["dirty"], bool)
    assert provenance()["environment"]["torch"] == torch.__version__


# --- aumento por velocidad y selección --------------------------------------


def dominant_bpm(y, fs=30.0):
    spectrum = np.abs(np.fft.rfft((y - y.mean()) * np.hanning(len(y)), n=8192))
    return 60 * np.fft.rfftfreq(8192, 1 / fs)[spectrum.argmax()]


@pytest.mark.parametrize("speed", [0.6, 1.4])
def test_speed_augmentation_scales_hr_of_video_and_label(speed):
    """Reproducir el tramo `speed` veces más rápido multiplica su HR por `speed`."""
    clip = make_clip(n=400, f_hz=1.5)  # 90 bpm
    aug = Augmenter({"speed_p": 1.0, "speed_range": [speed, speed]}, seed=0)
    ds = ChunkDataset([clip], chunk_length=128, stride=64, augmenter=aug)
    x, y = ds[0]
    assert x.shape == (3, 128, 16, 16) and y.shape == (128,)
    assert dominant_bpm(y.numpy()) == pytest.approx(90 * speed, abs=3)
    green = x[1].mean(dim=(1, 2)).numpy()  # el video cambia de velocidad junto con la PPG
    assert dominant_bpm(green) == pytest.approx(90 * speed, abs=3)


def test_speed_augmentation_falls_back_when_window_does_not_fit():
    clip = make_clip(n=130, f_hz=1.5)  # 1,4 x 128 = 179 frames > 130
    aug = Augmenter({"speed_p": 1.0, "speed_range": [1.4, 1.4]}, seed=0)
    _, y = ChunkDataset([clip], chunk_length=128, stride=64, augmenter=aug)[0]
    assert dominant_bpm(y.numpy()) == pytest.approx(90, abs=3)


def test_speed_augmentation_disabled_by_default_and_validates_range():
    assert Augmenter({}, seed=0).speed_factor() == 1.0
    with pytest.raises(ValueError):
        Augmenter({"speed_range": [1.4, 0.6]}, seed=0)


def test_interpolate_frames_exact_and_midpoint():
    frames = np.stack([np.full((2, 2, 3), v, dtype=np.uint8) for v in (0, 100, 200)])
    out = interpolate_frames(frames, np.array([0.0, 0.5, 1.0, 2.0]))
    np.testing.assert_allclose(out[:, 0, 0, 0], [0, 50, 100, 200])


def test_signal_losses_identical_and_inverted():
    t = np.arange(256) / 30
    y = np.sin(2 * np.pi * 1.2 * t)
    np.testing.assert_allclose(signal_losses(y, y, 64), 0, atol=1e-9)
    np.testing.assert_allclose(signal_losses(-y, y, 64), 2, atol=1e-9)
    assert len(signal_losses(y[:100], y[:100], 64)) == 1


def test_epochs_since_best():
    history = [{"val_loss": v} for v in (0.5, 0.3, 0.4, 0.35)]
    assert _epochs_since_best(history, "val_loss") == 2
    assert _epochs_since_best([], "val_loss") == 0
    assert _epochs_since_best([{"val_mae": 1.0}], "val_loss") == 0  # historial viejo sin val_loss


def test_invalid_selection_criterion_raises(tmp_path):
    cfg = tiny_cfg(tmp_path, 1)
    cfg["train"]["select_by"] = "val_pearson"
    with pytest.raises(ValueError, match="select_by"):
        train(
            cfg,
            EVAL_CFG,
            [make_clip(n=200)],
            [make_clip(n=400)],
            tmp_path / "r",
            log=lambda m: None,
        )

"""Tests del modelo de DL: arquitectura, pérdida, datos, inferencia y entrenamiento."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rppg.data.ubfc import UBFCrPPG  # noqa: E402
from rppg.dl.dataset import (  # noqa: E402
    Augmenter,
    ChunkDataset,
    SubjectClip,
    load_subject,
    standardize_frames,
)
from rppg.dl.infer import predict_signal  # noqa: E402
from rppg.dl.losses import neg_pearson  # noqa: E402
from rppg.dl.physnet import PhysNet  # noqa: E402
from rppg.dl.train import load_model, train  # noqa: E402
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
def test_predict_signal_covers_every_frame(n):
    clip = make_clip(n=n)
    out = predict_signal(GreenProbe(), clip.frames, chunk_length=32)
    assert out.shape == (n,) and np.isfinite(out).all()
    assert np.corrcoef(out, clip.ppg)[0, 1] > 0.9


def test_predict_signal_rejects_short_video():
    with pytest.raises(ValueError):
        predict_signal(GreenProbe(), make_clip(n=20).frames, chunk_length=32)


# --- entrenamiento -----------------------------------------------------------


def tiny_cfg(tmp_path, epochs):
    return {
        "seed": 0,
        "run_name": "tiny",
        "data": {"chunk_length": 32, "train_stride": 16, "input_size": 16},
        "augment": {"flip_p": 0.5, "gain_drift": 0.05, "noise_std": 0.005},
        "train": {
            "epochs": epochs,
            "batch_size": 4,
            "lr": 1e-3,
            "weight_decay": 0.0,
            "num_workers": 0,
            "amp": False,
            "device": "cpu",
        },
    }


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

    model, cfg = load_model(run / "best.pt")
    assert cfg["data"]["chunk_length"] == 32 and model.frames == 32

"""Entrenamiento de PhysNet con checkpoints reanudables (mitigación de R2).

Por época se guardan:
- `last.pt`: modelo, optimizador, escalador de AMP, estado de los generadores
  aleatorios e historial. Permite retomar exactamente donde se cortó la sesión.
- `best.pt`: el modelo con menor MAE de HR en validación (protocolo común).
- `history.csv`: pérdidas y métricas por época.
"""

import random
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rppg.dl.dataset import Augmenter, ChunkDataset, SubjectClip
from rppg.dl.infer import predict_signal
from rppg.dl.losses import neg_pearson
from rppg.dl.physnet import PhysNet
from rppg.eval.protocol import compare_hr, summarize_windows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (estado global, por si lo usa una dependencia)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def evaluate_clips(
    model: torch.nn.Module,
    clips: Sequence[SubjectClip],
    chunk_length: int,
    eval_cfg: Mapping[str, Any],
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predice cada video completo y lo evalúa con el protocolo común."""
    windows = []
    for clip in clips:
        pulse = predict_signal(model, clip.frames, chunk_length, device)
        windows.append(compare_hr(clip.subject, clip.t, pulse, clip.t, clip.ppg, eval_cfg))
    windows = pd.concat(windows, ignore_index=True)
    return windows, summarize_windows(windows)


def _rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),  # noqa: NPY002
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])  # noqa: NPY002
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def train(
    cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    train_clips: Sequence[SubjectClip],
    val_clips: Sequence[SubjectClip],
    run_dir: str | Path,
    resume: bool = False,
    log: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Entrena y devuelve el historial. Con resume=True retoma desde last.pt."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_cfg, train_cfg = cfg["data"], cfg["train"]
    seed_everything(cfg["seed"])
    device = resolve_device(train_cfg["device"])
    use_amp = bool(train_cfg["amp"]) and device.type == "cuda"

    augmenter = Augmenter(cfg.get("augment", {}), seed=cfg["seed"])
    dataset = ChunkDataset(
        train_clips, data_cfg["chunk_length"], data_cfg["train_stride"], augmenter
    )
    if len(dataset) == 0:
        raise ValueError("Training set has no valid chunks")
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        drop_last=len(dataset) > train_cfg["batch_size"],
        generator=torch.Generator().manual_seed(cfg["seed"]),
    )

    model = PhysNet(frames=data_cfg["chunk_length"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"]
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history: list[dict[str, float]] = []
    best_mae, start_epoch = float("inf"), 0

    last_path = run_dir / "last.pt"
    if resume and last_path.is_file():
        ckpt = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng"])
        loader.generator.set_state(ckpt["loader_rng"])
        augmenter.rng.bit_generator.state = ckpt["augment_rng"]
        history, best_mae, start_epoch = ckpt["history"], ckpt["best_mae"], ckpt["epoch"] + 1
        log(f"Resumed from epoch {ckpt['epoch'] + 1} (best val MAE {best_mae:.2f} bpm)")

    log(
        f"Device: {device} | AMP: {use_amp} | train chunks: {len(dataset)} | "
        f"val subjects: {len(val_clips)}"
    )
    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                pred = model(x)
            loss = neg_pearson(pred.float(), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        _, val_summary = evaluate_clips(
            model, val_clips, data_cfg["chunk_length"], eval_cfg, device
        )
        val = val_summary.loc["ALL"]
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_mae": float(val["mae"]),
            "val_rmse": float(val["rmse"]),
            "val_pearson": float(val["pearson"]),
        }
        history.append(row)
        improved = row["val_mae"] < best_mae
        if improved:
            best_mae = row["val_mae"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_mae": best_mae,
                    "config": dict(cfg),
                },
                run_dir / "best.pt",
            )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": _rng_state(),
                "loader_rng": loader.generator.get_state(),
                "augment_rng": augmenter.rng.bit_generator.state,
                "history": history,
                "best_mae": best_mae,
                "epoch": epoch,
                "config": dict(cfg),
            },
            last_path,
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(
            f"epoch {epoch + 1:3d}/{train_cfg['epochs']} | loss {row['train_loss']:.3f} | "
            f"val MAE {row['val_mae']:.2f} bpm | val r {row['val_pearson']:.2f}"
            f"{'  *' if improved else ''}"
        )
    return pd.DataFrame(history)


def load_model(checkpoint: str | Path, device: str | torch.device = "cpu") -> tuple[PhysNet, dict]:
    """Carga un checkpoint (best.pt o last.pt) y devuelve el modelo y su config."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PhysNet(frames=cfg["data"]["chunk_length"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg

"""Entrenamiento de PhysNet con checkpoints reanudables (mitigación de R2).

Por época se guardan:
- `last.pt`: modelo, optimizador, escalador de AMP, estado de los generadores
  aleatorios e historial. Permite retomar exactamente donde se cortó la sesión.
- `best.pt`: el modelo de la mejor época según `train.select_by`: `val_loss`
  (1 - Pearson de la señal en validación) o `val_mae` (MAE de HR, protocolo común).
- `history.csv`: pérdidas y métricas por época, con el commit que entrenó cada una.
- `run_info.json`: procedencia de cada sesión de la corrida (commit, cambios sin
  commitear, entorno, épocas cubiertas). Una corrida retomada en Colab puede
  tener varias sesiones; si usaron código distinto, queda registrado.
"""

import json
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
from rppg.utils.provenance import provenance


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (estado global, por si lo usa una dependencia)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def signal_losses(pulse: np.ndarray, ppg: np.ndarray, chunk_length: int) -> np.ndarray:
    """1 - Pearson entre señal predicha y PPG, por tramo sin solapamiento.

    Es la misma pérdida del entrenamiento, medida sobre el video completo. No
    depende de la estimación espectral de HR, así que no la afectan los errores
    de armónico y es más estable para elegir la mejor época.
    """
    losses = []
    for s in range(0, len(pulse) - chunk_length + 1, chunk_length):
        p, y = pulse[s : s + chunk_length], ppg[s : s + chunk_length]
        if np.std(p) > 0 and np.std(y) > 0:
            losses.append(1.0 - float(np.corrcoef(p, y)[0, 1]))
    return np.array(losses)


def inference_options(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Opciones de `predict_signal` según la sección `infer` de la config.

    Las configs anteriores a la v0.1.7 (y los checkpoints entrenados con ellas)
    no tienen esa sección: se usan tramos sin solapamiento y sin espejado, que
    era el único modo de inferencia.
    """
    infer = cfg.get("infer") or {}
    return {
        "stride": infer.get("stride") or cfg["data"]["chunk_length"],
        "flip_average": bool(infer.get("flip_average", False)),
    }


def evaluate_clips(
    model: torch.nn.Module,
    clips: Sequence[SubjectClip],
    chunk_length: int,
    eval_cfg: Mapping[str, Any],
    device: torch.device,
    stride: int | None = None,
    flip_average: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predice cada video completo y lo evalúa con el protocolo común.

    El resumen agrega la columna `signal_loss` (1 - Pearson por tramo): por
    sujeto y, en la fila "ALL", promediada sobre todos los tramos. `stride` y
    `flip_average` se pasan a `predict_signal`; los valores por defecto son los
    del entrenamiento.
    """
    windows, losses = [], {}
    for clip in clips:
        pulse = predict_signal(
            model, clip.frames, chunk_length, device, stride=stride, flip_average=flip_average
        )
        windows.append(compare_hr(clip.subject, clip.t, pulse, clip.t, clip.ppg, eval_cfg))
        losses[clip.subject] = signal_losses(pulse, clip.ppg, chunk_length)
    windows = pd.concat(windows, ignore_index=True)
    summary = summarize_windows(windows)
    summary["signal_loss"] = [
        float(np.concatenate(list(losses.values())).mean())
        if subject == "ALL"
        else float(losses[subject].mean())
        for subject in summary.index
    ]
    return windows, summary


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
    history: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    select_by = train_cfg.get("select_by", "val_mae")
    if select_by not in ("val_loss", "val_mae"):
        raise ValueError(f"train.select_by must be 'val_loss' or 'val_mae', got {select_by!r}")
    patience = train_cfg.get("early_stopping_patience")
    best_score, start_epoch = float("inf"), 0

    last_path = run_dir / "last.pt"
    if resume and last_path.is_file():
        # Se carga en CPU: los estados de los generadores aleatorios tienen que
        # ser ByteTensor de CPU. load_state_dict mueve modelo y optimizador al
        # dispositivo de los parámetros.
        ckpt = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        _set_rng_state(ckpt["rng"])
        loader.generator.set_state(ckpt["loader_rng"])
        augmenter.rng.bit_generator.state = ckpt["augment_rng"]
        history, start_epoch = ckpt["history"], ckpt["epoch"] + 1
        best_score = ckpt.get("best_score", ckpt.get("best_mae"))  # checkpoints v0.1.2
        sessions = ckpt.get("sessions", [])
        log(f"Resumed from epoch {start_epoch} (best {select_by} {best_score:.3f})")

    session = {**provenance(), "start_epoch": start_epoch + 1, "resumed": start_epoch > 0}
    sessions.append(session)
    git = session["git"]
    log(f"Code: {git['describe'] or 'unknown (no git)'} ({git['branch']})")
    if git["dirty"]:
        log("WARNING: uncommitted changes; this run cannot be traced to a single commit")
    if len(sessions) > 1 and sessions[-2]["git"]["commit"] != git["commit"]:
        log(
            f"WARNING: resuming with different code ({sessions[-2]['git']['short']} -> "
            f"{git['short']}); see run_info.json"
        )
    _write_run_info(run_dir, cfg, sessions)

    log(
        f"Device: {device} | AMP: {use_amp} | train chunks: {len(dataset)} | "
        f"val subjects: {len(val_clips)} | select by: {select_by} | "
        f"early stopping: {patience or 'off'}"
    )
    for epoch in range(start_epoch, train_cfg["epochs"]):
        if patience and _epochs_since_best(history, select_by) >= patience:
            log(f"Early stopping: no {select_by} improvement in {patience} epochs")
            break
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
            model, val_clips, data_cfg["chunk_length"], eval_cfg, device, **inference_options(cfg)
        )
        val = val_summary.loc["ALL"]
        row = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean(losses)),
            "val_mae": float(val["mae"]),
            "val_rmse": float(val["rmse"]),
            "val_pearson": float(val["pearson"]),
            "val_loss": float(val["signal_loss"]),
            "commit": f"{git['short'] or 'unknown'}{'-dirty' if git['dirty'] else ''}",
        }
        history.append(row)
        improved = row[select_by] < best_score
        if improved:
            best_score = row[select_by]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "selected_by": select_by,
                    "val_loss": row["val_loss"],
                    "val_mae": row["val_mae"],
                    "config": dict(cfg),
                    "provenance": session,
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
                "best_score": best_score,
                "epoch": epoch,
                "config": dict(cfg),
                "sessions": sessions,
            },
            last_path,
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        log(
            f"epoch {epoch + 1:3d}/{train_cfg['epochs']} | loss {row['train_loss']:.3f} | "
            f"val loss {row['val_loss']:.3f} | val MAE {row['val_mae']:.2f} bpm | "
            f"val r {row['val_pearson']:.2f}"
            f"{'  *' if improved else ''}"
        )
    return pd.DataFrame(history)


def _epochs_since_best(history: Sequence[Mapping[str, Any]], metric: str) -> int:
    """Épocas transcurridas desde la mejor según `metric` (0 si la última es la mejor)."""
    if not history:
        return 0
    values = [row.get(metric, np.inf) for row in history]
    return len(values) - 1 - int(np.argmin(values))


def _write_run_info(run_dir: Path, cfg: Mapping[str, Any], sessions: list[dict]) -> None:
    info = {"run_name": cfg.get("run_name"), "config": dict(cfg), "sessions": sessions}
    (run_dir / "run_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def load_model(
    checkpoint: str | Path, device: str | torch.device = "cpu"
) -> tuple[PhysNet, dict, dict]:
    """Carga un checkpoint (best.pt o last.pt).

    Devuelve el modelo, su config y metadatos: época, MAE de validación y
    procedencia (commit con que se entrenó). Los checkpoints anteriores a la
    v0.1.2 no tienen procedencia; ese campo queda en None.
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PhysNet(frames=cfg["data"]["chunk_length"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    provenance = ckpt.get("provenance") or (ckpt.get("sessions") or [None])[-1]
    meta = {
        "epoch": ckpt["epoch"] + 1,
        "selected_by": ckpt.get("selected_by", "val_mae"),
        "val_loss": ckpt.get("val_loss"),
        "val_mae": ckpt.get("val_mae", ckpt.get("best_mae")),
        "provenance": provenance,
    }
    return model, cfg, meta

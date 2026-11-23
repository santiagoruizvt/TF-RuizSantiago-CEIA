"""Ensamble de modelos PhysNet entrenados con distintas semillas (tarea 3.4).

Cada modelo predice la señal de un tramo; cada salida se estandariza (media 0,
desvío 1 a lo largo del tiempo, porque la escala de cada modelo es arbitraria)
y se promedian. Como el ensamble es un `nn.Module` con la misma entrada y salida
que PhysNet, la inferencia sobre el video completo (`predict_signal`, con
tramos solapados) y la evaluación (`evaluate_clips`) no cambian.

Promediar semillas reduce la dependencia del resultado respecto de la
inicialización: en validación, con HR bajas, una semilla caía en el armónico
y otra no (ver RELEASE_NOTES, v0.1.11).
"""

from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from rppg.dl.train import inference_options, load_model


class EnsembleModel(nn.Module):
    """Promedio de las salidas estandarizadas de varios modelos: (B, 3, T, H, W) -> (B, T)."""

    def __init__(self, members: Sequence[nn.Module]) -> None:
        super().__init__()
        if not members:
            raise ValueError("An ensemble needs at least one model")
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = torch.stack([m(x).float() for m in self.members])  # (M, B, T)
        outs = (outs - outs.mean(dim=-1, keepdim=True)) / (outs.std(dim=-1, keepdim=True) + 1e-6)
        return outs.mean(dim=0)


def load_ensemble(
    checkpoints: Sequence[str | Path], device: str | torch.device = "cpu"
) -> tuple[EnsembleModel, dict, list[dict]]:
    """Carga varios checkpoints como un ensamble.

    Devuelve el ensamble, la config del primer modelo y los metadatos de cada
    uno. Todos tienen que compartir lo que define la entrada y la inferencia:
    tamaño de los recortes, largo del tramo y opciones de `infer`.
    """
    loaded = [load_model(c, device) for c in checkpoints]
    cfg = loaded[0][1]
    reference = _compatibility_key(cfg)
    for path, (_, other, _) in zip(checkpoints, loaded, strict=True):
        if _compatibility_key(other) != reference:
            raise ValueError(f"{path} is not compatible with {checkpoints[0]}: {reference}")
    model = EnsembleModel([m for m, _, _ in loaded]).to(device).eval()
    metas = [
        {"checkpoint": str(c), **meta} for c, (_, _, meta) in zip(checkpoints, loaded, strict=True)
    ]
    return model, cfg, metas


def _compatibility_key(cfg: dict) -> tuple:
    data = cfg["data"]
    infer = inference_options(cfg)
    return data["input_size"], data["chunk_length"], infer["stride"], infer["flip_average"]

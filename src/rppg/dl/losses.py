"""Pérdidas para entrenar sobre la señal temporal."""

import torch


def neg_pearson(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """1 - correlación de Pearson, promediada sobre el batch.

    pred y target: (B, T). Vale 0 si las señales están perfectamente
    correlacionadas, 1 si no están correlacionadas y 2 si están invertidas. Es
    invariante a la escala y al offset de cada señal, que en rPPG no tienen
    significado físico. Equivale a `Neg_Pearson` del repositorio de PhysNet,
    vectorizada.
    """
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    num = (pred * target).sum(dim=1)
    den = torch.sqrt((pred**2).sum(dim=1) * (target**2).sum(dim=1) + eps)
    return (1 - num / den).mean()

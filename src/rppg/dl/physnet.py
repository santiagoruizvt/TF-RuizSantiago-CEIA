"""PhysNet (Yu, Li & Zhao, BMVC 2019): CNN espacio-temporal 3D encoder-decoder.

Arquitectura tomada sin cambios del repositorio original del autor
(https://github.com/ZitongYu/PhysNet, `PhysNetED_BMVC.py`, licencia MIT). Es la
misma red que redistribuye rPPG-Toolbox; se toma del original para no heredar la
licencia RAIL de rPPG-Toolbox (ver RELEASE_NOTES). El autor pide citar el
trabajo y usarlo solo con fines de investigación.

Cambios respecto del original: formato, nombres en inglés de las variables
internas, y `forward` devuelve solo la señal rPPG (el original devuelve además
mapas intermedios para visualización).

----------------------------------------------------------------------------
MIT License

Copyright (c) 2019 Zitong Yu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
----------------------------------------------------------------------------
"""

import torch
from torch import nn


def _conv_block(c_in: int, c_out: int, kernel, padding) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel, stride=1, padding=padding),
        nn.BatchNorm3d(c_out),
        nn.ReLU(inplace=True),
    )


def _upsample_block() -> nn.Sequential:
    # Duplica la dimensión temporal; deja la espacial igual.
    return nn.Sequential(
        nn.ConvTranspose3d(64, 64, kernel_size=[4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
        nn.BatchNorm3d(64),
        nn.ELU(),
    )


class PhysNet(nn.Module):
    """Entrada (B, 3, T, H, W) -> señal rPPG (B, T).

    T debe ser múltiplo de 4 (dos reducciones temporales a la mitad) e igual a
    `frames`. H y W deben ser múltiplos de 16 (cuatro reducciones espaciales).
    """

    def __init__(self, frames: int = 128) -> None:
        super().__init__()
        self.frames = frames
        self.conv1 = _conv_block(3, 16, [1, 5, 5], [0, 2, 2])
        self.conv2 = _conv_block(16, 32, [3, 3, 3], 1)
        self.conv3 = _conv_block(32, 64, [3, 3, 3], 1)
        self.conv4 = _conv_block(64, 64, [3, 3, 3], 1)
        self.conv5 = _conv_block(64, 64, [3, 3, 3], 1)
        self.conv6 = _conv_block(64, 64, [3, 3, 3], 1)
        self.conv7 = _conv_block(64, 64, [3, 3, 3], 1)
        self.conv8 = _conv_block(64, 64, [3, 3, 3], 1)
        self.conv9 = _conv_block(64, 64, [3, 3, 3], 1)
        self.upsample1 = _upsample_block()
        self.upsample2 = _upsample_block()
        self.conv10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        self.pool_spatial = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.pool_spatiotemporal = nn.MaxPool3d((2, 2, 2), stride=2)
        self.pool_global_spatial = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[2]
        if length != self.frames:
            raise ValueError(f"Expected {self.frames} frames, got {length}")
        x = self.pool_spatial(self.conv1(x))  # (16, T, H/2, W/2)
        x = self.conv3(self.conv2(x))  # (64, T, H/2, W/2)
        x = self.pool_spatiotemporal(x)  # (64, T/2, H/4, W/4)
        x = self.conv5(self.conv4(x))
        x = self.pool_spatiotemporal(x)  # (64, T/4, H/8, W/8)
        x = self.conv7(self.conv6(x))
        x = self.pool_spatial(x)  # (64, T/4, H/16, W/16)
        x = self.conv9(self.conv8(x))
        x = self.upsample2(self.upsample1(x))  # (64, T, H/16, W/16)
        x = self.conv10(self.pool_global_spatial(x))  # (1, T, 1, 1)
        return x.view(-1, length)

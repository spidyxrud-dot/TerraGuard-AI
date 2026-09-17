"""Siamese U-Net for bi-temporal change detection (Step 3.3).

Architecture (weight-shared encoder, FC-Siam style):

    before [B,4,H,W] ─┐
                       ├─► SAME encoder ─► feature pyramids (per date)
    after  [B,4,H,W] ─┘
                       │
        bottleneck: concat(before, after, |before - after|) ─► 1x1x? fusion
                       │
             U-Net decoder, skips = concat(before_feats, after_feats)
                       │
              logits [B, 1, H, W]  ──sigmoid──►  change probability

The encoder module instance is applied to both dates - Siamese behaviour is a
consequence of calling the *same* weights twice, not of duplicating layers.

The network is fully convolutional and pads internally (reflect) so any spatial size
>= 2**depth works - necessary because the TerraGuard inference grid is 1025 x 1025,
which a 4-level U-Net cannot consume raw. Output is cropped back to the input extent.

`forward` returns raw logits; sigmoid / thresholding live in :meth:`probability` and
:meth:`change_mask` so the decision threshold stays explicit and configurable instead
of being baked into the graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class SiameseUNetConfig:
    """Shape-defining knobs of the Siamese U-Net (recorded in model metadata)."""

    in_channels: int = 4
    """Input channels per date: 4 = [B02, B03, B04, B08], 3 = LEVIR-CD RGB contract."""

    base_channels: int = 16
    """Channels of the first encoder level; doubles per level."""

    depth: int = 4
    """Number of encoder levels (last one is the bottleneck). Input is padded to a
    multiple of ``2 ** depth``."""

    out_channels: int = 1
    """Output channels (change logits)."""

    def __post_init__(self) -> None:
        if self.in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {self.in_channels}")
        if self.base_channels < 1:
            raise ValueError(f"base_channels must be >= 1, got {self.base_channels}")
        if not 1 <= self.depth <= 6:
            raise ValueError(f"depth must be in [1, 6], got {self.depth}")

    @property
    def input_divisor(self) -> int:
        return 2 ** self.depth

    def to_dict(self) -> dict:
        return {"in_channels": self.in_channels, "base_channels": self.base_channels,
                "depth": self.depth, "out_channels": self.out_channels,
                "input_divisor": self.input_divisor}


class DoubleConv(nn.Sequential):
    """(Conv3x3 -> BN -> ReLU) x 2 with 'same' padding."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


def _pad_to_multiple(images: torch.Tensor, divisor: int) -> tuple[torch.Tensor, int, int]:
    """Reflect-pad the bottom/right so H and W become multiples of ``divisor``."""
    height, width = images.shape[-2:]
    pad_h = (divisor - height % divisor) % divisor
    pad_w = (divisor - width % divisor) % divisor
    if pad_h >= height or pad_w >= width:
        raise ValueError(f"input {height}x{width} is too small for divisor {divisor} "
                         f"(tiles smaller than {divisor} px are unsupported)")
    if pad_h == 0 and pad_w == 0:
        return images, 0, 0
    padded = F.pad(images, (0, pad_w, 0, pad_h), mode="reflect")
    return padded, pad_h, pad_w


class SiameseUNet(nn.Module):
    """Weight-shared Siamese encoder + U-Net decoder over a bi-temporal pair.

    Input : ``before [B, C, H, W]`` and ``after [B, C, H, W]`` (C = config.in_channels)
    Output: change logits ``[B, 1, H, W]`` (apply :meth:`probability` / :meth:`change_mask`)
    """

    def __init__(self, config: SiameseUNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or SiameseUNetConfig()
        cfg = self.config

        self.encoder = nn.ModuleList()
        channels = cfg.in_channels
        for level in range(cfg.depth):
            out_channels = cfg.base_channels * 2 ** level
            self.encoder.append(DoubleConv(channels, out_channels))
            channels = out_channels
        self.bottleneck_channels = channels
        self.pool = nn.MaxPool2d(2)

        # fusion of both dates' bottleneck features + their absolute difference
        self.fusion = DoubleConv(3 * self.bottleneck_channels, self.bottleneck_channels)

        self.up_samples = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for level in range(cfg.depth - 1, 0, -1):
            up_channels = cfg.base_channels * 2 ** (level - 1)
            self.up_samples.append(nn.ConvTranspose2d(channels, up_channels, kernel_size=2,
                                                      stride=2))
            # skips carry both dates (before + after) and the up-sampled tensor has
            # up_channels: 3 * up_channels in total
            self.decoder.append(DoubleConv(3 * up_channels, up_channels))
            channels = up_channels

        self.head = nn.Conv2d(cfg.base_channels, cfg.out_channels, kernel_size=1)

    # -- forward ----------------------------------------------------------------
    def _encode(self, images: torch.Tensor) -> list[torch.Tensor]:
        features = []
        current = images
        for block in self.encoder:
            current = block(current)
            features.append(current)
            current = self.pool(current)
        return features

    def forward(self, before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
        self.validate_inputs(before, after)
        divisor = self.config.input_divisor
        before_p, pad_h, pad_w = _pad_to_multiple(before, divisor)
        after_p, _, _ = _pad_to_multiple(after, divisor)

        before_feats = self._encode(before_p)
        after_feats = self._encode(after_p)

        bottleneck_b, bottleneck_a = before_feats[-1], after_feats[-1]
        fused = self.fusion(torch.cat([bottleneck_b, bottleneck_a,
                                       (bottleneck_b - bottleneck_a).abs()], dim=1))

        current = fused
        for up, block, skip_b, skip_a in zip(self.up_samples, self.decoder,
                                             reversed(before_feats[:-1]),
                                             reversed(after_feats[:-1])):
            current = up(current)
            current = block(torch.cat([skip_b, skip_a, current], dim=1))

        logits = self.head(current)
        height, width = before.shape[-2:]
        return logits[:, :, :height, :width]

    # -- output helpers -----------------------------------------------------------
    @staticmethod
    def probability(logits: torch.Tensor) -> torch.Tensor:
        """Change probability ``[B, 1, H, W]`` in [0, 1] from raw logits."""
        return torch.sigmoid(logits)

    @staticmethod
    def change_mask(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary change mask ``[B, 1, H, W]`` (bool) at the given probability threshold."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        return torch.sigmoid(logits) > threshold

    # -- validation / metadata ------------------------------------------------------
    def validate_inputs(self, before: torch.Tensor, after: torch.Tensor) -> None:
        if before.shape != after.shape:
            raise ValueError(f"before {tuple(before.shape)} and after {tuple(after.shape)} "
                             "must have identical shapes")
        if before.dim() != 4:
            raise ValueError(f"expected [B, C, H, W] tensors, got {tuple(before.shape)}")
        if before.shape[1] != self.config.in_channels:
            raise ValueError(f"expected {self.config.in_channels} input channels "
                             f"(config.in_channels), got {before.shape[1]}")
        if before.shape[-1] < self.config.input_divisor or \
                before.shape[-2] < self.config.input_divisor:
            raise ValueError(f"spatial size {tuple(before.shape[-2:])} is below the minimum "
                             f"{self.config.input_divisor}px for depth "
                             f"{self.config.depth}")

    def summary(self) -> dict:
        """Model facts for ``models/siamese_unet.json`` metadata."""
        return {"model": type(self).__name__,
                "config": self.config.to_dict(),
                "parameters": sum(parameter.numel() for parameter in self.parameters()),
                "trainable_parameters": sum(parameter.numel() for parameter in
                                            self.parameters() if parameter.requires_grad)}


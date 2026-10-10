"""Detection adaptation of GlassNet's dense contrast and context/channel attention.

Source: Lin et al., CVPR 2021, Rich Context Aggregation with Reflection Prior
for Glass Surface Detection (user-provided code.zip/model.py).
Original copyright (c) 2021 Jiaying Lin and Rynson W.H. Lau.
See GSD_LICENSE.txt for the retained source notice and conditions.
"""

import torch
from torch import nn

from .conv import Conv


class GSDContextAttention(nn.Module):
    """Weight six contiguous contrast groups and channels within each group."""

    def __init__(self, channels, reduction=4, contexts=6):
        super().__init__()
        if channels % contexts:
            raise ValueError('channels must be divisible by contexts')
        self.group_channels = channels // contexts
        hidden = contexts * max(1, self.group_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.context = nn.Sequential(
            nn.Conv2d(channels, max(1, channels // 2), 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // 2), contexts, 1, bias=False),
            nn.Sigmoid(),
        )
        self.channel = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, groups=contexts, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, groups=contexts, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = self.pool(x)
        context = self.context(pooled).repeat_interleave(self.group_channels, dim=1)
        return x * context * self.channel(pooled)


class GSDResidualContrast(nn.Module):
    """Four dilated branches, six pairwise contrasts, attention and residual.

    Unlike the original segmentation module, restore C channels and initialize
    the residual gain to zero to preserve the pretrained detector at startup.
    """

    def __init__(self, channels):
        super().__init__()
        branch_channels = max(4, ((channels // 8 + 3) // 4) * 4)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, branch_channels, 3, padding=rate, dilation=rate),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for rate in (1, 2, 4, 8)
        )
        self.attention = GSDContextAttention(6 * branch_channels)
        self.project = nn.Conv2d(6 * branch_channels, channels, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        contrasts = torch.cat(
            [features[i] - features[j] for i in range(4) for j in range(i + 1, 4)], dim=1
        )
        return x + self.gain * self.project(self.attention(contrasts))


class GSDContrastConv(Conv):
    """Conv-compatible projection followed by residual contrast enhancement.

    Keep conv/bn state-dict keys so baseline projection weights still load.
    Both forward paths retain attention when Ultralytics fuses Conv and BN.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__(c1, c2, k, s, p, g, d, act)
        self.contrast = GSDResidualContrast(c2)

    def forward(self, x):
        return self.contrast(super().forward(x))

    def forward_fuse(self, x):
        return self.contrast(super().forward_fuse(x))

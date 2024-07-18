import torch
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()
        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )
        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        x = x + self.token_mix(x)
        x = x + self.channel_mix(x)
        return x


class MLPMixer(nn.Module):
    def __init__(self,
                 in_channels=3,
                 dim=512,
                 patch_size=4,
                 image_size=32,
                 depth=8,
                 token_dim=256,
                 channel_dim=2048,
                 num_classes=10):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (image_size // patch_size) ** 2
        self.layers = torch.nn.ModuleList([nn.Conv2d(in_channels, dim, patch_size, patch_size),
                                           Rearrange('b c h w -> b (h w) c')])

        for _ in range(depth):
            self.layers.append(MixerBlock(dim, self.num_patch, token_dim, channel_dim))
        self.layers.append(nn.LayerNorm(dim))
        self.layers.append(nn.Linear(dim, num_classes))

    def forward(self, x, start=0, end=None):
        if end is None:
            end = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            if start <= i <= end:
                if i == len(self.layers) - 2:
                    x = layer(x)
                    x = x.mean(dim=1)
                else:
                    x = layer(x)
        return x

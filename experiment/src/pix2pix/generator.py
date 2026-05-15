import torch
import torch.nn as nn
from .convblock import ConvBlock


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, dim=64):
        super().__init__()

        self.down_layers = nn.ModuleList(
            [
                ConvBlock(in_channels, dim, normalize=False, leaky=True),  # L1
                ConvBlock(dim, dim * 2, normalize=True, leaky=True),  # L2
                ConvBlock(dim * 2, dim * 4, normalize=True, leaky=True),  # L3
                ConvBlock(dim * 4, dim * 8, normalize=True, leaky=True),  # L4
                ConvBlock(
                    dim * 8, dim * 8, normalize=True, leaky=True, drop=True
                ),  # L5
                ConvBlock(
                    dim * 8, dim * 8, normalize=True, leaky=True, drop=True
                ),  # L6
                ConvBlock(dim * 8, dim * 8, normalize=True, leaky=True),  # L7
                ConvBlock(
                    dim * 8, dim * 8, normalize=False, leaky=True
                ),  # L8 (Bottleneck)
            ]
        )

        self.up_layers = nn.ModuleList(
            [
                ConvBlock(
                    dim * 8, dim * 8, upsample=True, normalize=True, drop=True
                ),  # L9
                ConvBlock(
                    dim * 16, dim * 8, upsample=True, normalize=True, drop=True
                ),  # L10
                ConvBlock(dim * 16, dim * 8, upsample=True, normalize=True),  # L11
                ConvBlock(dim * 16, dim * 8, upsample=True, normalize=True),  # L12
                ConvBlock(dim * 16, dim * 4, upsample=True, normalize=True),  # L13
                ConvBlock(dim * 8, dim * 2, upsample=True, normalize=True),  # L14
                ConvBlock(dim * 4, dim, upsample=True, normalize=True),  # L15
            ]
        )

        self.final_up = nn.ConvTranspose2d(
            dim * 2, out_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        skips = []

        for i, down in enumerate(self.down_layers):
            x = down(x)
            if i < len(self.down_layers) - 1:
                skips.append(x)

        skips = skips[::-1]

        for i, up in enumerate(self.up_layers):
            x = up(x)
            x = torch.cat([skips[i], x], dim=1)

        x = self.final_up(x)
        return torch.tanh(x)

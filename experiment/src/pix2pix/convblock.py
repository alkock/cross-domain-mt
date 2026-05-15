import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        upsample=False,
        drop=True,
        normalize=True,
        leaky=True,
    ):
        super().__init__()

        self.drop = drop
        self.normalize = normalize
        self.leaky = leaky

        if not upsample:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not normalize,
            )
        else:
            self.conv = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=not normalize,
            )

        if normalize:
            self.norm = nn.InstanceNorm2d(out_channels)

        if drop:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        if self.normalize:
            x = self.norm(x)
        if self.drop:
            x = self.dropout(x)
        if self.leaky:
            x = F.leaky_relu(x, 0.2)
        else:
            x = F.relu(x)
        return x

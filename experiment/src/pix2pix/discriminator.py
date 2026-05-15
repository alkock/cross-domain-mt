import torch
import torch.nn as nn
from .convblock import ConvBlock


class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.layer1 = ConvBlock(in_channels * 2, 64, normalize=False)
        self.layer2 = ConvBlock(64, 128)
        self.layer3 = ConvBlock(128, 256)
        self.layer4 = ConvBlock(256, 512, stride=1)
        self.layer5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, src_image, trg_image):
        x = torch.cat([src_image, trg_image], dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

import torch
import torch.nn as nn
from generator_blocks import *

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_residuals=9):
        super().__init__()
        self.initial = G_InitialBlock(in_channels, out_channels)
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1),
                ConvBlock(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(out_channels*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(out_channels*4, out_channels*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                ConvBlock(out_channels*2, out_channels*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(out_channels*1, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))



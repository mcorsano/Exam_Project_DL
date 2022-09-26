import torch
import torch.nn as nn
from generator_blocks import *



class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()

        layers = list((
            G_InitialBlock(in_channels, out_channels),
            G_DownBlock(out_channels, out_channels*2, kernel_size=3, stride=2, padding=1),
            G_DownBlock(out_channels*2, out_channels*4, kernel_size=3, stride=2, padding=1),
            ResidualBlock(out_channels*4),
            ResidualBlock(out_channels*4),
            ResidualBlock(out_channels*4),
            ResidualBlock(out_channels*4),
            ResidualBlock(out_channels*4),
            ResidualBlock(out_channels*4),
            ResidualBlock(out_channels*4),
            ResidualBlock(out_channels*4),
            ResidualBlock(out_channels*4),
            G_UpBlock(out_channels*4, out_channels*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            G_UpBlock(out_channels*2, out_channels*1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Conv2d(out_channels*1, in_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")
        ))

        self.mod = nn.Sequential(*layers)

    def forward(self, x):
        return torch.tanh(self.mod(x))

import torch
import torch.nn as nn
from generator_blocks import *



class Generator(nn.Module):

    def __init__(self, channel_noise, img_channels, features_g):
        super().__init__()
        layers = list((
            G_Block(in_channels=channel_noise, out_channels=features_g*16, kernel_size=4, stride=1, padding=0),
            G_Block(in_channels=features_g*16, out_channels=features_g*8, kernel_size=4, stride=2, padding=1),
            G_Block(in_channels=features_g*8, out_channels=features_g*4, kernel_size=4, stride=2, padding=1),
            G_Block(in_channels=features_g*4, out_channels=features_g*2, kernel_size=4, stride=2, padding=1),
            G_FinalBlock(in_channels=features_g*2, out_channels=img_channels)
        ))

        self.mod = nn.Sequential(*layers)

    def forward(self, x):
        return self.mod(x)


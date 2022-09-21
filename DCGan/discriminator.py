import torch
import torch.nn as nn
from discriminator_blocks import *



class Discriminator(nn.Module):

    def __init__(self, img_channels, out_channels):
        super().__init__()
        layers = list((
            D_InitialBlock(in_channels=img_channels, out_channels=out_channels),
            D_Block(in_channels=out_channels, out_channels=out_channels*2),
            D_Block(in_channels=out_channels*2, out_channels=out_channels*4),
            D_Block(in_channels=out_channels*4, out_channels=out_channels*8),
            D_FinalBlock(in_channels=out_channels*8)
        ))

        self.mod = nn.Sequential(*layers)

    def forward(self, x):
        return self.mod(x)
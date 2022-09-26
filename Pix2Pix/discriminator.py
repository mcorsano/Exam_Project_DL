import torch
import torch.nn as nn
from discriminator_blocks import *

  
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        layers = list((
            D_InitialBlock(in_channels=3*2, out_channels=64, stride=2),
            D_Block(in_channels=64, out_channels=128, stride=2),
            D_Block(in_channels=128, out_channels=256, stride=2),
            D_Block(in_channels=256, out_channels=512, stride=1),
            D_FinalBlock(in_channels=512, out_channels=1, stride=1))
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        # y can be either fake or real.
        # is task of the discriminator to tell it
        x = torch.cat([x,y], dim=1)
        return self.model(x)



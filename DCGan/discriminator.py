import torch
import torch.nn as nn
from discriminator_blocks import *



class Discriminator(nn.Module):

    def __init__(self, img_channels, feature_d):
        super().__init__()
        layers = list((                                                          # input: N x img_channels x 64 x 64
            D_InitialBlock(in_channels=img_channels, out_channels=feature_d),    # 32 x 32
            D_Block(in_channels=feature_d, out_channels=feature_d*2),            # 16 X 16
            D_Block(in_channels=feature_d*2, out_channels=feature_d*4),          # 8 X 8
            D_Block(in_channels=feature_d*4, out_channels=feature_d*8),          # 4 X 4
            D_FinalBlock(in_channels=feature_d*8)                                # 1 X 1 (real image or not)
        ))

        self.mod = nn.Sequential(*layers)

    def forward(self, x):
        return self.mod(x)
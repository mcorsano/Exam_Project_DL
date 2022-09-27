import torch
import torch.nn as nn
from generator_blocks import *



class Generator(nn.Module):

    def __init__(self, z_dim, img_channels, features_gen):
        super().__init__()
        layers = list((                                                                                                 # input: N x z_dim x 1 x 1
            G_Block(in_channels=z_dim, out_channels=features_gen*16, kernel_size=4, stride=1, padding=0),               # N x features_gen*16 x 4 x 4
            G_Block(in_channels=features_gen*16, out_channels=features_gen*8, kernel_size=4, stride=2, padding=1),      # N x features_gen*8 x 8 x 8
            G_Block(in_channels=features_gen*8, out_channels=features_gen*4, kernel_size=4, stride=2, padding=1),       # N x features_gen*4 x 16 x 16
            G_Block(in_channels=features_gen*4, out_channels=features_gen*2, kernel_size=4, stride=2, padding=1),       # N x features_gen*2 x 32 x 32
            G_FinalBlock(in_channels=features_gen*2, out_channels=img_channels)                                         # N x 1 x 64 x 64
        ))

        self.mod = nn.Sequential(*layers)

    def forward(self, data):
        return self.mod(data)


import torch
import torch.nn as nn
from discriminator_blocks import *



class Discriminator(nn.Module):

    def __init__(self, img_channels, features_disc):
        super().__init__()
        layers = list((                                                                  # input: N x img_channels x 64 x 64
            D_InitialBlock(in_channels=img_channels, out_channels=features_disc),        # 32 x 32
            D_Block(in_channels=features_disc, out_channels=features_disc*2),            # 16 X 16
            D_Block(in_channels=features_disc*2, out_channels=features_disc*4),          # 8 X 8
            D_Block(in_channels=features_disc*4, out_channels=features_disc*8),          # 4 X 4
            D_FinalBlock(in_channels=features_disc*8)                                    # 1 X 1 (real image or not)
        ))

        self.mod = nn.Sequential(*layers)

    def forward(self, data):
        return self.mod(data)
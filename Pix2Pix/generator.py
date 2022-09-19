import torch
import torch.nn as nn
from generator_blocks import *


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.down1 = G_NonBatchBlock(in_channels=3, out_channels=64, activation="LeakyReLU")
        self.down2 = G_DownBlock(in_channels=64, out_channels=128)
        self.down3 = G_DownBlock(in_channels=128, out_channels=256)
        self.down4 = G_DownBlock(in_channels=256, out_channels=512)
        self.down5 = G_DownBlock(in_channels=512, out_channels=512)
        self.down6 = G_DownBlock(in_channels=512, out_channels=512)
        self.down7 = G_DownBlock(in_channels=512, out_channels=512)
        self.steady = G_NonBatchBlock(in_channels=512, out_channels=512, activation="ReLU")
        self.up1 = G_UpBlock(in_channels=512, out_channels=512, dropout=True)
        self.up2 = G_UpBlock(in_channels=512*2, out_channels=512, dropout=True)
        self.up3 = G_UpBlock(in_channels=512*2, out_channels=512, dropout=True)
        self.up4 = G_UpBlock(in_channels=512*2, out_channels=512, dropout=False)
        self.up5 = G_UpBlock(in_channels=512*2, out_channels=256, dropout=False)
        self.up6 = G_UpBlock(in_channels=256*2, out_channels=128, dropout=False)
        self.up7 = G_UpBlock(in_channels=128*2, out_channels=64, dropout=False)
        self.up8 = G_FinalBlock(in_channels=128, out_channels=3)


    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)
        down5 = self.down5(down4)
        down6 = self.down6(down5)
        down7 = self.down7(down6)
        steady = self.steady(down7)
        up1 = self.up1(steady)
        up2 = self.up2(torch.cat([up1, down7], 1))
        up3 = self.up3(torch.cat([up2, down6], 1))
        up4 = self.up4(torch.cat([up3, down5], 1))
        up5 = self.up5(torch.cat([up4, down4], 1))
        up6 = self.up6(torch.cat([up5, down3], 1))
        up7 = self.up7(torch.cat([up6, down2], 1))
        return self.up8(torch.cat([up7, down1], 1))

import torch
import torch.nn as nn
from generator_blocks import *


class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.down_block1 = G_NonBatchBlock(in_channels=3, out_channels=64, activation="LeakyReLU")      # 128
        self.down_block2 = G_DownBlock(in_channels=64, out_channels=128)                                # 64
        self.down_block3 = G_DownBlock(in_channels=128, out_channels=256)                               # 32
        self.down_block4 = G_DownBlock(in_channels=256, out_channels=512)                               # 16
        self.down_block5 = G_DownBlock(in_channels=512, out_channels=512)                               # 8
        self.down_block6 = G_DownBlock(in_channels=512, out_channels=512)                               # 4
        self.down_block7 = G_DownBlock(in_channels=512, out_channels=512)                               # 2
        self.steady_block = G_NonBatchBlock(in_channels=512, out_channels=512, activation="ReLU")       # 1 x 1
        self.up_block1 = G_UpBlock(in_channels=512, out_channels=512, dropout=True)
        self.up_block2 = G_UpBlock(in_channels=512*2, out_channels=512, dropout=True)
        self.up_block3 = G_UpBlock(in_channels=512*2, out_channels=512, dropout=True)
        self.up_block4 = G_UpBlock(in_channels=512*2, out_channels=512, dropout=False)
        self.up_block5 = G_UpBlock(in_channels=512*2, out_channels=256, dropout=False)
        self.up_block6 = G_UpBlock(in_channels=256*2, out_channels=128, dropout=False)
        self.up_block7 = G_UpBlock(in_channels=128*2, out_channels=64, dropout=False)
        self.up_block8 = G_FinalBlock(in_channels=128, out_channels=3)


    def forward(self, data):
        down_block1 = self.down_block1(data)            
        down_block2 = self.down_block2(down_block1)        
        down_block3 = self.down_block3(down_block2)        
        down_block4 = self.down_block4(down_block3)        
        down_block5 = self.down_block5(down_block4)        
        down_block6 = self.down_block6(down_block5)        
        down_block7 = self.down_block7(down_block6)        
        steady_block = self.steady_block(down_block7)      
        up_block1 = self.up_block1(steady_block)
        up_block2 = self.up_block2(torch.cat([up_block1, down_block7], 1))
        up_block3 = self.up_block3(torch.cat([up_block2, down_block6], 1))
        up_block4 = self.up_block4(torch.cat([up_block3, down_block5], 1))
        up_block5 = self.up_block5(torch.cat([up_block4, down_block4], 1))
        up_block6 = self.up_block6(torch.cat([up_block5, down_block3], 1))
        up_block7 = self.up_block7(torch.cat([up_block6, down_block2], 1))
        up_block8 = self.up_block8(torch.cat([up_block7, down_block1], 1))
        return up_block8

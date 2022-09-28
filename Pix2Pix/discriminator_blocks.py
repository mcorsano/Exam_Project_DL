import torch
import torch.nn as nn



class D_InitialBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

    def forward(self, data):
        return self.mod(data)



class D_Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, data):
        return self.mod(data)



class D_FinalBlock(nn.Module):

    def __init__(self, in_channels, out_channels=1, stride=1):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect"),
            nn.Sigmoid()
        )
        
    def forward(self, data):
        return self.mod(data)
import torch
import torch.nn as nn



class D_InitialBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.mod(x)



class D_Block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),     # because of the batch normalization, we set bias false in Conv layer above
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.mod(x)



class D_FinalBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.mod(x)
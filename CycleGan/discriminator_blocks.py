import torch 
import torch.nn as nn



class D_InitialBlock(nn.Module):
    def  __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=4, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.mod(x)



class D_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride, kernel_size=4, padding=1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels), 
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.mod(x)



class D_FinalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.mod = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x):
        return self.mod(x)


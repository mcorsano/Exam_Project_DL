import torch 
import torch.nn as nn



class G_InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.mod(x)



class G_DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_ReLU=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if use_ReLU else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)



class G_UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_ReLU=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() if use_ReLU else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)



class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            G_DownBlock(channels, channels, kernel_size=3, padding=1),
            G_DownBlock(channels, channels, use_ReLU=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

import torch
import torch.nn as nn



class D_InitialBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.mod(x)



class D_Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.mod(x)



class D_FinalBlock(nn.Module):

    def __init__(self, in_channels, out_channels=1, stride=1):
        super().__init__()
        self.mod = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, padding_mode="reflect")

    def forward(self, x):
        return self.mod(x)


    
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
        # is task of the disciminator to tell it
        x = torch.cat([x,y], dim=1)
        return self.model(x)

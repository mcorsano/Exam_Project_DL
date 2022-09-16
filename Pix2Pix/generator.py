from stringprep import in_table_a1
import torch
import torch.nn as nn


class G_NonBatchBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, activation="ReLU"):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2) if activation=="LeakyRelU" else nn.ReLU()
        )

    def forward(self, x):
        return self.mod(x)



class G_DownBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.mod(x)



class G_UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels, dropout):
        super().__init__()
        self.use_dropout = dropout
        self.dropout = nn.Dropout(0.5)
        self.mod = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.mod(x)
        return self.dropout(x) if self.use_dropout else x




class G_FinalBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mod = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # each pixel will have a value in [-1,1]
        )

    def forward(self, x):
        return self.mod(x)



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

import torch
import torch.nn as nn



class G_NonBatchBlock(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, activation="ReLU"):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.LeakyReLU(0.2) if activation=="LeakyRelU" else nn.ReLU()
        )

    def forward(self, data):
        return self.mod(data)



class G_DownBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, data):
        return self.mod(data)



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
    
    def forward(self, data):
        data = self.mod(data)
        return self.dropout(data) if self.use_dropout else data



class G_FinalBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.mod = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # each pixel will have a value in [-1,1]
        )

    def forward(self, data):
        return self.mod(data)
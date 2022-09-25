import torch
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 1),   # fake=0, real=1
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.mod(x)
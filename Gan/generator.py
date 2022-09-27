import torch
import torch.nn as nn



class Generator(nn.Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.mod = nn.Sequential(
            nn.Linear(z_dim, 256),    # z_dim is the dimension of the latent noise
            nn.LeakyReLU(0.01),
            nn.Linear(256, img_dim),
            nn.Tanh(),  # we'll normalize inputs to [-1, 1] so we need outputs (px values) in [-1, 1]
        )

    def forward(self, data):
        return self.mod(data)
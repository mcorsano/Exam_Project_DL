from ctypes import util
from typing_extensions import dataclass_transform
import torch
import utilities
import torch.nn as nn
import torch.optim as optim
from dataset import Pix2PixDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import matplotlib.pyplot as plt



def train_epoch(discriminator, generator, dataloader, discriminator_optimizer, generator_optimizer, L1_loss, bce):

    loop = tqdm(dataloader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(utilities.DEVICE)
        y = y.to(utilities.DEVICE)

        # Train Discriminator
        y_fake = generator(x)
        D_real = discriminator(x, y)
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake = discriminator(x, y_fake.detach())
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2

        discriminator.zero_grad()
        D_loss.backward()
        discriminator_optimizer.step()

        # Train generator
        D_fake = discriminator(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        L1 = L1_loss(y_fake, y) * utilities.L1_LAMBDA
        G_loss = G_fake_loss + L1

        generator_optimizer.zero_grad()
        G_loss.backward()
        generator_optimizer.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )



    print("done")



def train_model(discriminator, generator, train_dataLoader, validation_dataLoader, D_optimizer, G_optimizer, L1_LOSS, BCE):
        for epoch in range(utilities.NUM_EPOCHS):
            train_epoch(discriminator, generator, train_dataLoader, D_optimizer, G_optimizer, L1_LOSS, BCE)

            utilities.save_images(generator, validation_dataLoader, epoch, folder="validations")
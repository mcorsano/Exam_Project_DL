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



def train_model(dataloader, generator, discriminator, generator_optimizer, discriminator_optimizer, L1_loss, bce):

    loop = tqdm(dataloader, leave=True)

    for epoch in range(utilities.NUM_EPOCHS):
        for batch_idx, (real, gen_real) in enumerate(loop):
            real = real.to(utilities.DEVICE)
            gen_real = gen_real.to(utilities.DEVICE)
            gen_fake = generator(real)

            # Train Discriminator
            D_real = discriminator(real, gen_real)
            loss_D_real = bce(D_real, torch.ones_like(D_real))

            D_fake = discriminator(real, gen_fake)
            loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))

            discriminator_loss = (loss_D_real + loss_D_fake) / 2

            discriminator.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            # Train generator
            D_fake = discriminator(real, gen_fake)
            loss_G_fake = bce(D_fake, torch.ones_like(D_fake))
            
            L1 = L1_loss(gen_fake, gen_real) * utilities.L1_LAMBDA   # in the paper they added the L1 loss to the generator loss
            generator_loss = loss_G_fake + L1

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            if batch_idx % 10 == 0:

                print(
                    f"Epoch [{epoch}/{utilities.NUM_EPOCHS}] Batch {batch_idx}/{len(dataloader)} \
                        Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}"
                )

                utilities.save_images(generator, dataloader, epoch, folder="validations")

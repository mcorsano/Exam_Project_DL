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



def train_model(train_dataLoader, val_dataLoader, generator, discriminator, generator_optimizer, discriminator_optimizer, L1_loss, bce):

    loop = tqdm(train_dataLoader, leave=True)

    for epoch in range(utilities.NUM_EPOCHS):
        for batch_index, (real_img, target_img) in enumerate(loop):
            real_img = real_img.to(utilities.DEVICE)
            target_img = target_img.to(utilities.DEVICE)
            fake_img = generator(real_img)

            # Train Discriminator
            D_real = discriminator(real_img, target_img)
            loss_D_real = bce(D_real, torch.ones_like(D_real))

            D_fake = discriminator(real_img, fake_img)
            loss_D_fake = bce(D_fake, torch.zeros_like(D_fake))

            discriminator_loss = (loss_D_real + loss_D_fake) / 2

            discriminator.zero_grad()
            discriminator_loss.backward(retain_graph=True)
            discriminator_optimizer.step()

            # Train generator
            D_fake = discriminator(real_img, fake_img)
            loss_G_fake = bce(D_fake, torch.ones_like(D_fake))
            
            L1 = L1_loss(fake_img, target_img)*utilities.L1_LAMBDA   # in the paper they added the L1 loss to the generator loss
            generator_loss = loss_G_fake + L1

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            if batch_index % 10 == 0:

                print(
                    f"Epoch [{epoch}/{utilities.NUM_EPOCHS}] Batch {batch_index}/{len(train_dataLoader)} \
                        Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}"
                )

                utilities.save_images(generator, val_dataLoader, epoch, folder="validations")

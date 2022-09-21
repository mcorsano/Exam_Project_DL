import torch
from dataset import HorseZebraDataset
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
import utilities



def train_epoch(H_generator, H_discriminator, Z_generator, Z_discriminator, dataLoader, discriminator_optimizer, generator_optimizer, L1_loss, mse):
    H_reals = 0
    H_fakes = 0
    loop = tqdm(dataLoader, leave=True)

    for idx, (zebra, horse) in enumerate(loop):
        zebra = zebra.to(utilities.DEVICE)
        horse = horse.to(utilities.DEVICE)

        # Train discriminators
        fake_horse = H_generator(zebra)
        D_H_real = H_discriminator(horse)
        D_H_fake = H_discriminator(fake_horse.detach())
        H_reals += D_H_real.mean().item()
        H_fakes += D_H_fake.mean().item()
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = D_H_real_loss + D_H_fake_loss

        fake_zebra = Z_generator(horse)
        D_Z_real = Z_discriminator(zebra)
        D_Z_fake = Z_discriminator(fake_zebra.detach())
        D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
        D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
        D_Z_loss = D_Z_real_loss + D_Z_fake_loss

        # put it togethor
        D_loss = (D_H_loss + D_Z_loss)/2

        discriminator_optimizer.zero_grad()
        D_loss.backward()
        discriminator_optimizer.step()
        #discriminator_optimizer.update()

        # Train Generators H and Z
        # adversarial loss for both generators
        D_H_fake = H_discriminator(fake_horse)
        D_Z_fake = Z_discriminator(fake_zebra)
        loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
        loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

        # cycle loss
        cycle_zebra = Z_generator(fake_horse)
        cycle_horse = H_generator(fake_zebra)
        cycle_zebra_loss = L1_loss(zebra, cycle_zebra)
        cycle_horse_loss = L1_loss(horse, cycle_horse)

        # identity loss (remove these for efficiency if you set lambda_identity=0)
        identity_zebra = Z_generator(zebra)
        identity_horse = H_generator(horse)
        identity_zebra_loss = L1_loss(zebra, identity_zebra)
        identity_horse_loss = L1_loss(horse, identity_horse)

        # add all togethor
        G_loss = (
            loss_G_Z
            + loss_G_H
            + cycle_zebra_loss * utilities.LAMBDA_CYCLE
            + cycle_horse_loss * utilities.LAMBDA_CYCLE
            + identity_horse_loss * utilities.LAMBDA_IDENTITY
            + identity_zebra_loss * utilities.LAMBDA_IDENTITY
        )

        generator_optimizer.zero_grad()
        G_loss.backward()
        generator_optimizer.step()
        #generator_optimizer.update()

        if idx % 5 == 0:                                                                            #○ chenged from 200 to 5
            save_image(fake_horse*0.5+0.5, f"saved_images/horse_{idx}.png")
            save_image(fake_zebra*0.5+0.5, f"saved_images/zebra_{idx}.png")

        loop.set_postfix(H_real=H_reals/(idx+1), H_fake=H_fakes/(idx+1))
        
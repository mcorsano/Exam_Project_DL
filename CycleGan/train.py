import torch
from dataset import CycleGanDataset
import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator
import utilities



def train_model(y_generator, y_discriminator, x_generator, x_discriminator, dataLoader, val_dataLoader, discriminator_optimizer, generator_optimizer, L1_loss, mse):
    
    loop = tqdm(dataLoader, leave=True)
    
    for epoch in range(utilities.NUM_EPOCHS): 

        for batch_index, (x, y) in enumerate(loop):
            x = x.to(utilities.DEVICE)
            y = y.to(utilities.DEVICE)

            ### Train discriminators
            fake_y = y_generator(x)
            y_disc_real = y_discriminator(y)
            y_disc_fake = y_discriminator(fake_y.detach())
            loss_y_disc_real = mse(y_disc_real, torch.ones_like(y_disc_real))
            loss_y_disc_fake = mse(y_disc_fake, torch.zeros_like(y_disc_fake))
            loss_y_disc_total = loss_y_disc_real + loss_y_disc_fake

            fake_x = x_generator(y)
            x_disc_real = x_discriminator(x)
            x_disc_fake = x_discriminator(fake_x.detach())
            loss_x_disc_real = mse(x_disc_real, torch.ones_like(x_disc_real))
            loss_x_disc_fake = mse(x_disc_fake, torch.zeros_like(x_disc_fake))
            loss_x_disc_total = loss_x_disc_real + loss_x_disc_fake

            # discriminator loss
            discriminator_loss = loss_y_disc_total + loss_x_disc_total

            discriminator_optimizer.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()


            ### Train Generators 
            # adversarial loss
            y_disc_fake = y_discriminator(fake_y)
            x_disc_fake = x_discriminator(fake_x)
            loss_y_gen = mse(y_disc_fake, torch.ones_like(y_disc_fake))
            loss_x_gen = mse(x_disc_fake, torch.ones_like(x_disc_fake))

            # identity loss
            identity_loss_x = L1_loss(x, x_generator(x))
            identity_loss_y = L1_loss(y, y_generator(y))

            # cycle loss
            cycle_loss_x = L1_loss(x, x_generator(fake_y))
            cycle_loss_y = L1_loss(y, y_generator(fake_x))

            # generator loss
            generator_loss = (
                loss_x_gen
                + loss_y_gen
                + cycle_loss_x * utilities.LAMBDA_CYCLE_LOSS
                + cycle_loss_y * utilities.LAMBDA_CYCLE_LOSS
                + identity_loss_y * utilities.LAMBDA_IDENTITY_LOSS
                + identity_loss_x * utilities.LAMBDA_IDENTITY_LOSS
            )

            generator_optimizer.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            if batch_index % 2 == 0:         

                print(
                    f"Epoch [{epoch}/{utilities.NUM_EPOCHS}] Batch {batch_index}/{len(dataLoader)} \
                        Loss D: {discriminator_loss:.4f}, loss G: {generator_loss:.4f}"
                )
                
                utilities.save_images(x_generator, y_generator, val_dataLoader, epoch, folder="saved_images")

            
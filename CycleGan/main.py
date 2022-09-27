import torch
import torch.nn as nn
from train import train_model                   # prova -> train
from discriminator import Discriminator
from generator import Generator 
from torchvision.utils import save_image
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CycleGanDataset
from test import test_model
import utilities


def main():

    y_discriminator = Discriminator().to(utilities.DEVICE)
    y_generator = Generator().to(utilities.DEVICE)

    x_discriminator = Discriminator().to(utilities.DEVICE)
    x_generator = Generator().to(utilities.DEVICE)

    discriminator_optimizer = optim.Adam(list(y_discriminator.parameters()) + list(x_discriminator.parameters()), lr=utilities.LEARNING_RATE, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(list(x_generator.parameters()) + list(y_generator.parameters()), lr=utilities.LEARNING_RATE, betas=(0.5, 0.999))

    MSE = nn.MSELoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = CycleGanDataset(directory_y=utilities.TRAIN_DIR+"/horses", directory_x=utilities.TRAIN_DIR+"/zebras", transform=utilities.transforms)
    train_dataLoader = DataLoader(train_dataset, batch_size=utilities.BATCH_SIZE, shuffle=True, num_workers=utilities.NUM_WORKERS) 
    
    val_dataset = CycleGanDataset(directory_y=utilities.VAL_DIR+"/horses", directory_x=utilities.VAL_DIR+"/zebras", transform=utilities.transforms)
    val_dataLoader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    test_dataset = CycleGanDataset(directory_y=utilities.TEST_DIR+"/horses", directory_x=utilities.TEST_DIR+"/zebras", transform=utilities.transforms)
    test_dataLoader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    train_model(y_generator, y_discriminator, x_generator, x_discriminator, train_dataLoader, val_dataLoader, discriminator_optimizer, generator_optimizer, L1_LOSS, MSE)
    test_model(x_generator=x_generator, y_generator=y_generator, testLoader=test_dataLoader)

if __name__ == "__main__":
    main()
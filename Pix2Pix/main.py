from cgi import test
from ctypes import util
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
from train import train_model
from test import test_model





def main():
    discriminator = Discriminator().to(utilities.DEVICE)
    generator = Generator().to(utilities.DEVICE)

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=utilities.LEARNING_RATE, betas=(0.5, 0.999))
    generator_optimizer = optim.Adam(generator.parameters(), lr=utilities.LEARNING_RATE, betas=(0.5, 0.999))
    
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    train_dataset = Pix2PixDataset(dataset_dir=utilities.TRAIN_DIR)
    train_dataLoader = DataLoader(dataset=train_dataset, batch_size=utilities.BATCH_SIZE, shuffle=True, num_workers=utilities.NUM_WORKERS)

    validation_dataset = Pix2PixDataset(dataset_dir=utilities.VAL_DIR)
    validation_dataLoader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False)

    test_dataset = Pix2PixDataset(dataset_dir=utilities.TEST_DIR)
    test_dataLoader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    train_model(train_dataLoader, generator, discriminator, generator_optimizer, discriminator_optimizer, L1_LOSS, BCE)
    test_model(generator=generator, testLoader=test_dataLoader)    




if __name__ == "__main__":
    main()
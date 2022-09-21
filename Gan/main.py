import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import utilities
from discriminator import Discriminator
from generator import Generator
from train import train_model


def main():

    discriminator = Discriminator(utilities.IMAGE_DIM).to(utilities.DEVICE)
    generator = Generator(utilities.Z_DIM, utilities.IMAGE_DIM).to(utilities.DEVICE)

    dataset = datasets.MNIST(root="dataset/", transform=utilities.transforms, download=True)
    loader = DataLoader(dataset, batch_size=utilities.BATCH_SIZE, shuffle=True)
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=utilities.LEARNING_RATE)
    generator_optimizer = optim.Adam(generator.parameters(), lr=utilities.LEARNING_RATE)
    loss = nn.BCELoss()
 
    train_model(loader, generator, discriminator, generator_optimizer, discriminator_optimizer, loss)




if __name__== '__main__':
    main()






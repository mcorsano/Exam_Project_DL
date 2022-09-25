import torch
import torch.nn as nn
from discriminator import *
from generator import *
import torchvision.datasets as datasets
import utilities
from torch.utils.data import DataLoader
from train import *
import torch.optim as optim



def main():
    # remember to set IMAGE_CHANNELS to 1 when using MNIST dataset
    # dataset = datasets.MNIST(root="mnistDataset/", train=True, transform=utilities.transforms, download=True)
    # dataset = datasets.CelebA(root="celebaDataset/", transform=utilities.transforms, download=True)
    dataset = datasets.Flowers102(root="flowersDataset/", transform=utilities.transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=utilities.BATCH_SIZE, shuffle=True)
    
    generator = Generator(utilities.NOISE_DIM, utilities.IMAGE_CHANNELS, utilities.FEATURES_GEN).to(utilities.DEVICE)
    discriminator = Discriminator(utilities.IMAGE_CHANNELS, utilities.FEATURES_DISC).to(utilities.DEVICE)
    
    initialize_weights(generator)
    initialize_weights(discriminator)

    generator_optimizer = optim.Adam(generator.parameters(), lr=utilities.LEARNING_RATE, betas=(0.5, 0.999))
    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=utilities.LEARNING_RATE, betas=(0.5, 0.999))
    
    lossCriteria = nn.BCELoss()

    generator.train()
    discriminator.train()

    train_model(dataloader, generator, discriminator, generator_optimizer, discriminator_optimizer, lossCriteria)


if __name__ == '__main__':
    main()
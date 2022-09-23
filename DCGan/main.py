import torch
import torch.nn as nn
from discriminator import *
from generator import *
import torchvision.datasets as datasets
import utilities
from torch.utils.data import DataLoader
from train import *
import torch.optim as optim
# from dataset import PokemonDataset



def main():
    # If you train on MNIST, remember to set channels_img to 1
    dataset = datasets.MNIST(root="dataset/", train=True, transform=utilities.transforms, download=True)
    # dataset = PokemonDataset(dataset_dir=utilities.DATASET_DIR)

    # comment mnist above and uncomment below if train on CelebA
    #dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
    dataloader = DataLoader(dataset, batch_size=utilities.BATCH_SIZE, shuffle=True)
    gen = Generator(utilities.NOISE_DIM, utilities.CHANNELS_IMG, utilities.FEATURES_GEN).to(utilities.DEVICE)
    disc = Discriminator(utilities.CHANNELS_IMG, utilities.FEATURES_DISC).to(utilities.DEVICE)
    initialize_weights(gen)
    initialize_weights(disc)

    opt_gen = optim.Adam(gen.parameters(), lr=utilities.G_LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=utilities.D_LEARNING_RATE, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    # criterion = nn.BCEWithLogitsLoss()

    # writer_real = SummaryWriter(f"logs/real")
    # writer_fake = SummaryWriter(f"logs/fake")
    # step = 0

    gen.train()
    disc.train()

    train_model(dataloader, gen, disc, opt_gen, opt_disc, criterion)


if __name__ == '__main__':
    main()
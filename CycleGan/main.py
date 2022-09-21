import torch
import torch.nn as nn
from train import train_epoch
from discriminator import Discriminator
from generator import Generator 
from torchvision.utils import save_image
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import HorseZebraDataset
import utilities


def main():
    disc_H = Discriminator().to(utilities.DEVICE)
    disc_Z = Discriminator().to(utilities.DEVICE)
    gen_Z = Generator().to(utilities.DEVICE)
    gen_H = Generator().to(utilities.DEVICE)
    opt_disc = optim.Adam(
        list(disc_H.parameters()) + list(disc_Z.parameters()),
        lr=utilities.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_Z.parameters()) + list(gen_H.parameters()),
        lr=utilities.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    # dataset = HorseZebraDataset(
    #     root_horse=utilities.TRAIN_DIR+"/horses", root_zebra=utilities.TRAIN_DIR+"/zebras", transform=utilities.transforms
    # )
    # val_dataset = HorseZebraDataset(
    #    root_horse="cyclegan_test/horse1", root_zebra="cyclegan_test/zebra1", transform=utilities.transforms
    # )
    dataset = HorseZebraDataset(
        root_horse=utilities.TRAIN_DIR+"/horses", root_zebra=utilities.TRAIN_DIR+"/zebras", transform=utilities.transforms
    )
    val_dataset = HorseZebraDataset(
       root_horse=utilities.VAL_DIR+"/horses", root_zebra=utilities.VAL_DIR+"/zebras", transform=utilities.transforms
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=utilities.BATCH_SIZE,
        shuffle=True,
        num_workers=utilities.NUM_WORKERS,
        pin_memory=True
    )

    for epoch in range(utilities.NUM_EPOCHS):
        train_epoch(gen_H, disc_H, gen_Z, disc_Z, loader, opt_disc, opt_gen, L1, mse)



if __name__ == "__main__":
    main()
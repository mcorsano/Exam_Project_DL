from typing_extensions import dataclass_transform
import torch
import utilities
import torch.nn as nn
import torch.optim as optim
from dataset import Pix2PixDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from torchvision.utils import save_image



def test_model(generator, testLoader):
    real, gen_real = next(iter(testLoader))

    generator.eval()
    with torch.no_grad():
        it = 0
        for real, gen_real in testLoader:
            real = real.to(utilities.DEVICE)
            gen_real = gen_real.to(utilities.DEVICE)

            gen_fake = generator(real)
            save_image(real*0.5+0.5, "tests3" + f"/real_{it}.png")
            save_image(gen_fake*0.5+0.5, "tests3" + f"/gen_fake_{it}.png")
            it += 1


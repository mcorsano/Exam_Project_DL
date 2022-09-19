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



def test_model(generator, testLoader):
    x, y = next(iter(testLoader))
    x, y = x.to(utilities.DEVICE), y.to(utilities.DEVICE)

    generator.eval()
    with torch.no_grad():
        it = 0
        for x, y in testLoader:
            x = x.to(utilities.DEVICE)
            y = y.to(utilities.DEVICE)
            y_hat = generator(x)
            y_hat = y_hat * 0.5 + 0.5  # remove normalization #
            save_image(y_hat, "tests" + f"/y_gen_{it}.png")
            it += 1


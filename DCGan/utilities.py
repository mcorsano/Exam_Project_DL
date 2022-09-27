import torch
import torch.nn as nn
import torchvision.transforms as transforms



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
Z_DIM = 100
FEATURES_DISC = 64
FEATURES_GEN = 64
IMG_SIZE = 64
IMG_CHANNELS = 3
NUM_EPOCHS = 5000



transforms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(IMG_CHANNELS)], [0.5 for _ in range(IMG_CHANNELS)])
    ]
)

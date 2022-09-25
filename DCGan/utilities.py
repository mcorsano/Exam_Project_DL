import torch
import torch.nn as nn
import torchvision.transforms as transforms



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4  # could also use two lrs, one for gen and one for disc, but this works fine
BATCH_SIZE = 128
IMAGE_SIZE = 64
IMAGE_CHANNELS = 1
NOISE_DIM = 100
NUM_EPOCHS = 5000
FEATURES_DISC = 64
FEATURES_GEN = 64



transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(IMAGE_CHANNELS)], [0.5 for _ in range(IMAGE_CHANNELS)])
    ]
)

import torch
import torchvision.transforms as transforms



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 3e-4  # could also use two lrs, one for gen and one for disc, but this works fine
Z_DIM = 64
BATCH_SIZE = 32
IMAGE_DIM = 28*28*1   # 784
NUM_EPOCHS = 100



transforms = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ]
)
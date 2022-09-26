import torch
from torchvision.utils import save_image



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/maps/train"
VAL_DIR = "data/maps/val"
TEST_DIR = "data/maps/test"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 500



def save_images(generator, validationLoader, epoch, folder):
    real, gen_real = next(iter(validationLoader))
    real, gen_real = real.to(DEVICE), gen_real.to(DEVICE)

    generator.eval()
    with torch.no_grad():
        gen_fake = generator(real)
        save_image(real*0.5+0.5, folder + f"/real_{epoch}.png")
        save_image(gen_fake*0.5+0.5, folder + f"/gen_fake_{epoch}.png")
        if epoch == 1:
            save_image(gen_real*0.5+0.5, folder + f"/label_{epoch}.png")
    generator.train()
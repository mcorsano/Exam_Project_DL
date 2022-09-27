import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
VAL_DIR = "data/val"
TRAIN_DIR = "data/train"
TEST_DIR = "data/test"
BATCH_SIZE = 1
LAMBDA_IDENTITY_LOSS = 0.0
LAMBDA_CYCLE_LOSS = 10
NUM_WORKERS = 4
NUM_EPOCHS = 200



transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
     ],
    additional_targets={"image0": "image"},
)



def save_images(x_generator, y_generator, validationLoader, epoch, folder):
    x, y = next(iter(validationLoader))
    x, y = x.to(DEVICE), y.to(DEVICE)

    x_generator.eval()
    y_generator.eval()
    
    with torch.no_grad():
        fake_x = x_generator(y)
        fake_y = y_generator(x)

        save_image(fake_y*0.5+0.5, folder + f"/horse_fake_{epoch}.png")
        save_image(y*0.5+0.5, folder + f"/horse_{epoch}.png")
        save_image(fake_x*0.5+0.5, folder + f"/zebra_fake_{epoch}.png")
        save_image(x*0.5+0.5, folder + f"/zebra_{epoch}.png")
    x_generator.train()
    y_generator.train()

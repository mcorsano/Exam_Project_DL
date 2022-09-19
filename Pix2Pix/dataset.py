from hmac import trans_36
from statistics import mean
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


both_transform = A.Compose(
    [A.Resize(width=256, height=256),], 
    additional_targets={"image0": "image"},
)

input_img_transform_pipeline = A.Compose(
    [
        A.HorizontalFlip(always_apply=False, p=0.5),    # standard value
        A.ColorJitter(brightness=0.2),  # standard values
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ]
)

target_img_transform_pipeline = A.Compose(
    [
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2()
    ]
)



def transform(img_path):
    image = cv2.imread(img_path)
    # image = np.array(Image.open(img_path))

    input_img = image[:,:600,:]
    target_img = image[:,600:,:]
    augmentations = both_transform(image = input_img, image0=target_img)
    input_img = augmentations["image"]
    target_img = augmentations["image0"]

    input_img = input_img_transform_pipeline(image=input_img)["image"]   # [all_channels, :600 px in width, all px in height]
    target_img = target_img_transform_pipeline(image=target_img)["image"]     # [all_channels, 600: px in width, all px in height]
    return input_img, target_img


class Pix2PixDataset(Dataset):

    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir  # this is the directory of the dataset (e.g. "maps/train", "maps/test", ...)
        self.list_files = os.listdir(self.dataset_dir)  # list of all files present in the dataset_directory

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.dataset_dir, img_file)
        return transform(img_path)

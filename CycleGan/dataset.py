from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np



class CycleGanDataset(Dataset):
    def __init__(self, directory_x, directory_y, transform=None):
        self.directory_x = directory_x
        self.directory_y = directory_y
        self.transform = transform

        self.images_x = os.listdir(directory_x)
        self.images_y = os.listdir(directory_y)
        
        self.length_dataset = max(len(self.images_x), len(self.images_y)) # 1000, 1500
        self.len_x = len(self.images_x)
        self.len_y = len(self.images_y)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        image_x = self.images_x[index % self.len_x]        # we take the % bc datasets of x and y can be of different sizes 
        path_x = os.path.join(self.directory_x, image_x)
        image_x = np.array(Image.open(path_x).convert("RGB"))

        image_y = self.images_y[index % self.len_y]
        path_y = os.path.join(self.directory_y, image_y)
        image_y = np.array(Image.open(path_y).convert("RGB"))

        if self.transform:
            aug = self.transform(image=image_x, image0=image_y)
            image_x = aug["image"]
            image_y = aug["image0"]

        return image_x, image_y
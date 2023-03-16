# CNN model with custom data
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from skimage import io

# create annotations from the images
data_dir = 'cats_dogs/'
annotations = []
for file in os.listdir(data_dir):
    if 'cat' in file:
        annotations.append({'filename':file, 'label': 0})
    if 'dog' in file:
        annotations.append({'filename': file, 'label': 1})

df = pd.DataFrame(annotations)
df.to_csv('annotations.csv')


class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotation = pd.read_csv(csv_file)
        for col in self.annotation.columns:
            if 'Unnamed' in col:
                self.annotation.drop(col, inplace=True, axis=1)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        path = os.path.join(self.root_dir, self.annotation.iloc[index, 0])
        image = io.imread(path)
        label = torch.tensor(int(self.annotation.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return image, label




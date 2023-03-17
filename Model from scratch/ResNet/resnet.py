import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper parameters
num_classes = 1000
in_channels = 3
lr = 0.001
batch = 8
epochs = 1
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


class ResNet50(nn.Module): # layers : [3*3, 4*3, 6*3 ,3*3]
    def __init__(self,block, layers, image_channels, num_classes):
        super(ResNet50, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(in_channels=image_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)  # 112 * 112 * 64

        # resnet layers
        self.layer1 = self._make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_res_blocks, out_channels, stride):
        down_sample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels*4:
            down_sample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride), nn.BatchNorm2d(out_channels*4))

        layers.append(block(self.in_channels, out_channels, down_sample, stride))
        self.in_channels = out_channels * 4

        for i in range(num_res_blocks-1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


class res_block(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=None, stride=1):
        super(res_block, self).__init__()
        self.last_expansion = 4 * out_channels
        self.down_sample = down_sample
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=self.last_expansion, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.bn3 = nn.BatchNorm2d(self.last_expansion)
        self.relu = nn.ReLU()

    def forward(self, x):
        skip_con = x

        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        x = self.bn3(self.conv3(x))
        if self.down_sample is not None:
            skip_con = self.down_sample(skip_con)
        x += skip_con
        x = self.relu(x)
        return x


def create_ResNet(image_channels=3, num_classes=1000):
    return ResNet50(res_block, [3, 4, 6, 3], image_channels, num_classes)


model = create_ResNet().to(device)
x = torch.rand(4, 3, 224, 224).to(device)
print(model)
print(model(x).shape)
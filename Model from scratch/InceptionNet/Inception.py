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

# train_data = datasets.CIFAR10(root='../../datasets/', download=True, train=True, transform=transform)
# train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
# test_data = datasets.CIFAR10(root='../../datasets/', download=True, train=False, transform=transform)
# test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)


class InceptionNet(nn.Module):
    def __init__(self, in_channels=in_channels, num_classes=num_classes):
        super(InceptionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.inception3a = Inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.inception4a = Inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.inception5a = Inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_block(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=(7, 7), stride=(1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


class Inception_block(nn.Module):
    def __init__(self, input_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_block, self).__init__()
        self.branch1 = conv_block(input_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(input_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=(3, 3), padding=1),
        )
        self.branch3 = nn.Sequential(
            conv_block(input_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(input_channels, out_1x1pool, kernel_size=1),
        )

    def forward(self, x):
        # dim : N * channels * height * width
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class conv_block(nn.Module):
    def __init__(self, input_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=out_channels, **kwargs)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


model = InceptionNet().to(device)
x = torch.rand((4, 3, 224, 224)).to(device)
print(model(x).shape)
# torch.Size([4, 1000]) output shape for 1000 classes

# reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), padding=(1, 1),
                               stride=(1, 1))  # same convolutions
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.linear = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = fn.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = fn.relu(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.linear(x)
        return x


model = CNN()
x = torch.rand((64, 1, 28, 28))
print(model(x).shape)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Hyper parameters
in_channel = 1
num_class = 10
lr = 0.001
batch = 64
epochs = 3

# load dataset
train_data = datasets.MNIST(root='datasets/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_data = datasets.MNIST(root='datasets/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)

# Initialize network
model = CNN().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
op = optim.Adam(model.parameters(), lr=lr)

# train network
for e in range(epochs):
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        scores = model(x)
        loss = criterion(scores, y)

        op.zero_grad()
        loss.backward()

        op.step()


def accuracy_check(loader, model):
    if loader.dataset.train:
        print('checking accuracy on train data')
    else:
        print('checking accuracy on test data')
    num_correct = 0
    num_samples = 0
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()  # increase the number of correct predictions from the batch
            num_samples += preds.size(0)

    print(f"Accuracy is {(num_correct / num_samples) * 100}")


accuracy_check(train_loader, model)
accuracy_check(test_loader, model)

# output
# checking accuracy on train data
# Accuracy is 97.88333129882812
# checking accuracy on test data
# Accuracy is 97.98999786376953

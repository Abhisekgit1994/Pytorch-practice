import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create a fully connected network
class SimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):   # MNIST dataset 28*28 = 784
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = fn.relu(x)
        x = self.fc2(x)
        return x


model = SimpleNN(784, 10)
x = torch.rand((64,784))
print(model(x).shape)

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# Hyper parameters
input_size = 784
num_class = 10
lr = 0.001
batch = 64
epochs = 1

# load data
train_datasets = datasets.MNIST(root='dataset/', train=True,download=True, transform=transforms.ToTensor())
train_loader= DataLoader(train_datasets, batch_size=batch, shuffle=True)
test_datasets = datasets.MNIST(root='dataset/', train=False,download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_datasets, batch_size=batch, shuffle=True)

# initialize network
model = SimpleNN(input_size=input_size, num_classes=num_class).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# train network
for epoch in range(epochs):
    for batch_id, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        data = data.reshape(data.shape[0], -1)  # shaping to 784 dim
        pred = model(data)
        loss = criterion(pred, targets)

        # backward propagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent adam step
        optimizer.step()
    print('training completed')


# check accuracy on training and test on test set
def accuracy(loader, model):
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

            x = x.reshape(x.shape[0], -1)
            scores = model(x) # 64 * 10
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

    acc = (num_correct/num_samples)*100

    print(f"Accuracy is {(num_correct/num_samples)*100}")
    model.train()
    return acc


accuracy(train_loader, model)
accuracy(test_loader, model)

# output
# checking accuracy on train data
# Accuracy is 94.8983383178711
# checking accuracy on test data
# Accuracy is 94.48999786376953


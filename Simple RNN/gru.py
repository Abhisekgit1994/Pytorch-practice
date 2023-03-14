import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyper parameters
input_size = 28
seq_len = 28
num_layers = 2
hidden_size = 256
num_class = 10
lr = 0.001
batch = 64
epochs = 2


# Create a RNN
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, self.hidden_size, self.num_layers, batch_first=True)
        # N * time_seq * features
        self.fc = nn.Linear(self.hidden_size*seq_len, num_classes)

    def forward(self, x):
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h1)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


# load dataset
train_data = datasets.MNIST(root='../datasets/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_data = datasets.MNIST(root='../datasets/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)

# Initialize network
model = GRU(input_size, hidden_size, num_layers, num_class).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
op = optim.Adam(model.parameters(), lr=lr)

# train network
for e in range(epochs):
    for idx, (x, y) in enumerate(train_loader):
        x = x.to(device).squeeze(1)
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
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()  # increase the number of correct predictions from the batch
            num_samples += preds.size(0)

    print(f"Accuracy is {(num_correct / num_samples) * 100}")


print('********Printing accuracy for GRU model***********')
accuracy_check(train_loader, model)
accuracy_check(test_loader, model)

# ********Printing accuracy for GRU model***********
# checking accuracy on train data
# Accuracy is 98.61333465576172
# checking accuracy on test data
# Accuracy is 98.39999389648438
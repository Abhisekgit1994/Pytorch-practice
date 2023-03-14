# reference : https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
# reference : youtube.com/watch?v=jGst43P-TJA&list=PTachyonLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=6

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
D = 2  # bidirectional


# Create a bidirectional RNN
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_size, num_class):
        """

        :param input_size: input to the model
        :param hidden_size: hidden state features
        :param num_layers: number of lstm layers stacked
        :param num_class: output labels
        """
        super(BiLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch = batch_size
        self.num_classes = num_class
        self.b_lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * D, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        :param x: input
        :return: output of model
        """
        h0 = torch.zeros(self.num_layers * D, self.batch, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * D, self.batch, self.hidden_size).to(device)
        out, (hn, cn) = self.b_lstm(x, (h0, c0))
        hn = torch.cat((hn[-1], hn[-2]), dim=1)  # using last 2 hidden state to feed to linear layer as it is bidirectional as out.shape == (batch, hidden_size * D)
        out = self.fc(hn)
        return out


# load data
train_data = datasets.MNIST(root='../datasets/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_data = datasets.MNIST(root='../datasets/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)

# Initialize network
model = BiLSTM(input_size, hidden_size, num_layers,batch, num_class).to(device)

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
        if (idx + 1) % batch == 0:
            print(
                f"Epoch [{e + 1}/{epochs}], "
                f"Step [{idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )
        elif (idx + 1) == len(train_loader):
            print(
                f"Epoch [{e + 1}/{epochs}], "
                f"Step [{idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )


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


print('******** Printing accuracy for Bidirectional LSTM model ***********')
accuracy_check(train_loader, model)
accuracy_check(test_loader, model)

# Model progress and output for bidirectional LSTM
# Epoch [1/2], Step [64/938], Loss: 1.1996
# Epoch [1/2], Step [128/938], Loss: 0.5377
# Epoch [1/2], Step [192/938], Loss: 0.2803
# Epoch [1/2], Step [256/938], Loss: 0.3212
# Epoch [1/2], Step [320/938], Loss: 0.2201
# Epoch [1/2], Step [384/938], Loss: 0.1648
# Epoch [1/2], Step [448/938], Loss: 0.2547
# Epoch [1/2], Step [512/938], Loss: 0.0915
# Epoch [1/2], Step [576/938], Loss: 0.1543
# Epoch [1/2], Step [640/938], Loss: 0.1397
# Epoch [1/2], Step [704/938], Loss: 0.1492
# Epoch [1/2], Step [768/938], Loss: 0.0994
# Epoch [1/2], Step [832/938], Loss: 0.1220
# Epoch [1/2], Step [896/938], Loss: 0.0535
# Epoch [1/2], Step [938/938], Loss: 0.1159
# Epoch [2/2], Step [64/938], Loss: 0.1138
# Epoch [2/2], Step [128/938], Loss: 0.0659
# Epoch [2/2], Step [192/938], Loss: 0.0599
# Epoch [2/2], Step [256/938], Loss: 0.1399
# Epoch [2/2], Step [320/938], Loss: 0.0910
# Epoch [2/2], Step [384/938], Loss: 0.3076
# Epoch [2/2], Step [448/938], Loss: 0.1420
# Epoch [2/2], Step [512/938], Loss: 0.1811
# Epoch [2/2], Step [576/938], Loss: 0.0169
# Epoch [2/2], Step [640/938], Loss: 0.0376
# Epoch [2/2], Step [704/938], Loss: 0.1193
# Epoch [2/2], Step [768/938], Loss: 0.0192
# Epoch [2/2], Step [832/938], Loss: 0.0541
# Epoch [2/2], Step [896/938], Loss: 0.0629
# Epoch [2/2], Step [938/938], Loss: 0.1975
# ******** Printing accuracy for Bidirectional LSTM model ***********
# checking accuracy on train data
# Accuracy is 97.91333770751953
# checking accuracy on test data
# Accuracy is 97.83999633789062

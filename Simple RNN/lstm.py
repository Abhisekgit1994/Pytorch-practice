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
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        # N * time_seq * features
        self.fc = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        h1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c1 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h1, c1))
        out = self.fc(out[:, -1, :])  # using only the last hidden state output
        return out


# load dataset
train_data = datasets.MNIST(root='../datasets/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_data = datasets.MNIST(root='../datasets/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)

# Initialize network
model = LSTM(input_size, hidden_size, num_layers, num_class).to(device)

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
        elif (idx+1) == len(train_loader):
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


print('******** Printing accuracy for LSTM model ***********')
accuracy_check(train_loader, model)
accuracy_check(test_loader, model)

# **** model progress and output *********
# Epoch [1/2], Step [64/938], Loss: 0.9352
# Epoch [1/2], Step [128/938], Loss: 0.7253
# Epoch [1/2], Step [192/938], Loss: 0.5904
# Epoch [1/2], Step [256/938], Loss: 0.5197
# Epoch [1/2], Step [320/938], Loss: 0.1802
# Epoch [1/2], Step [384/938], Loss: 0.2368
# Epoch [1/2], Step [448/938], Loss: 0.1548
# Epoch [1/2], Step [512/938], Loss: 0.3303
# Epoch [1/2], Step [576/938], Loss: 0.0272
# Epoch [1/2], Step [640/938], Loss: 0.1656
# Epoch [1/2], Step [704/938], Loss: 0.0938
# Epoch [1/2], Step [768/938], Loss: 0.0546
# Epoch [1/2], Step [832/938], Loss: 0.1305
# Epoch [1/2], Step [896/938], Loss: 0.1241
# Epoch [1/2], Step [938/938], Loss: 0.1387
# Epoch [2/2], Step [64/938], Loss: 0.0620
# Epoch [2/2], Step [128/938], Loss: 0.1533
# Epoch [2/2], Step [192/938], Loss: 0.1051
# Epoch [2/2], Step [256/938], Loss: 0.0325
# Epoch [2/2], Step [320/938], Loss: 0.0213
# Epoch [2/2], Step [384/938], Loss: 0.1235
# Epoch [2/2], Step [448/938], Loss: 0.0987
# Epoch [2/2], Step [512/938], Loss: 0.0523
# Epoch [2/2], Step [576/938], Loss: 0.0467
# Epoch [2/2], Step [640/938], Loss: 0.1857
# Epoch [2/2], Step [704/938], Loss: 0.0270
# Epoch [2/2], Step [768/938], Loss: 0.0311
# Epoch [2/2], Step [832/938], Loss: 0.0750
# Epoch [2/2], Step [896/938], Loss: 0.0435
# Epoch [2/2], Step [938/938], Loss: 0.0170
# ******** Printing accuracy for LSTM model ***********
# checking accuracy on train data
# Accuracy is 98.2316665649414
# checking accuracy on test data
# Accuracy is 98.3499984741211
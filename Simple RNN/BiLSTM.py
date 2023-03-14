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
epochs = 4
D = 2  # bidirectional
load_model = True


# Create a bidirectional RNN
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,  num_class):
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
        self.num_classes = num_class
        self.b_lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size * D, self.num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: input
        :return: output of model
        """
        h0 = torch.zeros(self.num_layers * D, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * D, x.size(0), self.hidden_size).to(device)
        out, (hn, cn) = self.b_lstm(x, (h0, c0))
        hn = torch.cat((hn[-1], hn[-2]), dim=1)  # using last 2 hidden state to feed to linear layer as it is bidirectional as out.shape == (batch, hidden_size * D)
        out = self.fc(hn)
        return out


# save checkpoint
def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('Saving checkpoint......')
    torch.save(state, filename)


def load_checkpoint(checkpoint):
    print("loading checkpoint......")
    model.load_state_dict(checkpoint['state_dict'])
    op.load_state_dict(checkpoint['optimizer'])


# load data
train_data = datasets.MNIST(root='../datasets/', train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_data = datasets.MNIST(root='../datasets/', train=False, download=True, transform=transforms.ToTensor())
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)

# Initialize network
model = BiLSTM(input_size, hidden_size, num_layers, num_class).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
op = optim.Adam(model.parameters(), lr=lr)

# load a saved model
if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.tar'))


# train network
def train_network(num_epochs):
    for e in range(num_epochs):
        losses = []
        if e != 0 and e % 2 == 0:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': op.state_dict()}
            save_checkpoint(checkpoint)

        for idx, (x, y) in enumerate(train_loader):
            x = x.to(device).squeeze(1)
            y = y.to(device)

            scores = model(x)
            loss = criterion(scores, y)

            op.zero_grad()
            loss.backward()

            op.step()
            losses.append(loss.item())
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
        print("Loss at epoch", e+1, sum(losses)/len(losses))


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


print("****** training the model********")
train_network(epochs)
print('******** Printing accuracy for Bidirectional LSTM model ***********')
accuracy_check(train_loader, model)
accuracy_check(test_loader, model)

# Model progress and output for bidirectional LSTM
# loading checkpoint......
# ****** training the model********
# Epoch [1/4], Step [64/938], Loss: 0.2050
# Epoch [1/4], Step [128/938], Loss: 0.0840
# Epoch [1/4], Step [192/938], Loss: 0.1165
# Epoch [1/4], Step [256/938], Loss: 0.0084
# Epoch [1/4], Step [320/938], Loss: 0.0137
# Epoch [1/4], Step [384/938], Loss: 0.0427
# Epoch [1/4], Step [448/938], Loss: 0.0519
# Epoch [1/4], Step [512/938], Loss: 0.1486
# Epoch [1/4], Step [576/938], Loss: 0.0113
# Epoch [1/4], Step [640/938], Loss: 0.0627
# Epoch [1/4], Step [704/938], Loss: 0.0025
# Epoch [1/4], Step [768/938], Loss: 0.0208
# Epoch [1/4], Step [832/938], Loss: 0.0718
# Epoch [1/4], Step [896/938], Loss: 0.0086
# Epoch [1/4], Step [938/938], Loss: 0.1559
# Loss at epoch 1 0.0573388326972443
# Epoch [2/4], Step [64/938], Loss: 0.0580
# Epoch [2/4], Step [128/938], Loss: 0.0177
# Epoch [2/4], Step [192/938], Loss: 0.0262
# Epoch [2/4], Step [256/938], Loss: 0.0256
# Epoch [2/4], Step [320/938], Loss: 0.0441
# Epoch [2/4], Step [384/938], Loss: 0.0179
# Epoch [2/4], Step [448/938], Loss: 0.0229
# Epoch [2/4], Step [512/938], Loss: 0.1263
# Epoch [2/4], Step [576/938], Loss: 0.0227
# Epoch [2/4], Step [640/938], Loss: 0.0020
# Epoch [2/4], Step [704/938], Loss: 0.0118
# Epoch [2/4], Step [768/938], Loss: 0.0066
# Epoch [2/4], Step [832/938], Loss: 0.0045
# Epoch [2/4], Step [896/938], Loss: 0.0216
# Epoch [2/4], Step [938/938], Loss: 0.1338
# Loss at epoch 2 0.045341174875292725
# Saving checkpoint......
# Epoch [3/4], Step [64/938], Loss: 0.0072
# Epoch [3/4], Step [128/938], Loss: 0.0031
# Epoch [3/4], Step [192/938], Loss: 0.0532
# Epoch [3/4], Step [256/938], Loss: 0.0016
# Epoch [3/4], Step [320/938], Loss: 0.0411
# Epoch [3/4], Step [384/938], Loss: 0.0026
# Epoch [3/4], Step [448/938], Loss: 0.0066
# Epoch [3/4], Step [512/938], Loss: 0.0072
# Epoch [3/4], Step [576/938], Loss: 0.1365
# Epoch [3/4], Step [640/938], Loss: 0.0540
# Epoch [3/4], Step [704/938], Loss: 0.1321
# Epoch [3/4], Step [768/938], Loss: 0.0631
# Epoch [3/4], Step [832/938], Loss: 0.0016
# Epoch [3/4], Step [896/938], Loss: 0.0388
# Epoch [3/4], Step [938/938], Loss: 0.0581
# Loss at epoch 3 0.03608403073686824
# Epoch [4/4], Step [64/938], Loss: 0.0026
# Epoch [4/4], Step [128/938], Loss: 0.0268
# Epoch [4/4], Step [192/938], Loss: 0.0218
# Epoch [4/4], Step [256/938], Loss: 0.0911
# Epoch [4/4], Step [320/938], Loss: 0.0566
# Epoch [4/4], Step [384/938], Loss: 0.1525
# Epoch [4/4], Step [448/938], Loss: 0.0358
# Epoch [4/4], Step [512/938], Loss: 0.0213
# Epoch [4/4], Step [576/938], Loss: 0.0364
# Epoch [4/4], Step [640/938], Loss: 0.0277
# Epoch [4/4], Step [704/938], Loss: 0.0079
# Epoch [4/4], Step [768/938], Loss: 0.0029
# Epoch [4/4], Step [832/938], Loss: 0.2054
# Epoch [4/4], Step [896/938], Loss: 0.0029
# Epoch [4/4], Step [938/938], Loss: 0.0015
# Loss at epoch 4 0.031322088243948894
# ******** Printing accuracy for Bidirectional LSTM model ***********
# checking accuracy on train data
# Accuracy is 98.961669921875
# checking accuracy on test data
# Accuracy is 98.33999633789062

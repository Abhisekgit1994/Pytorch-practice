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
load_model = False


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
        hn = self.relu(hn)
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
# Epoch [1/4], Step [64/938], Loss: 1.0296
# Epoch [1/4], Step [128/938], Loss: 0.4420
# Epoch [1/4], Step [192/938], Loss: 0.3037
# Epoch [1/4], Step [256/938], Loss: 0.1739
# Epoch [1/4], Step [320/938], Loss: 0.2971
# Epoch [1/4], Step [384/938], Loss: 0.1947
# Epoch [1/4], Step [448/938], Loss: 0.2334
# Epoch [1/4], Step [512/938], Loss: 0.2162
# Epoch [1/4], Step [576/938], Loss: 0.1475
# Epoch [1/4], Step [640/938], Loss: 0.0740
# Epoch [1/4], Step [704/938], Loss: 0.1209
# Epoch [1/4], Step [768/938], Loss: 0.0282
# Epoch [1/4], Step [832/938], Loss: 0.2039
# Epoch [1/4], Step [896/938], Loss: 0.1897
# Epoch [1/4], Step [938/938], Loss: 0.0605
# Loss at epoch 1 0.34473418074447526
# Epoch [2/4], Step [64/938], Loss: 0.0763
# Epoch [2/4], Step [128/938], Loss: 0.0567
# Epoch [2/4], Step [192/938], Loss: 0.1380
# Epoch [2/4], Step [256/938], Loss: 0.0483
# Epoch [2/4], Step [320/938], Loss: 0.0146
# Epoch [2/4], Step [384/938], Loss: 0.0509
# Epoch [2/4], Step [448/938], Loss: 0.0704
# Epoch [2/4], Step [512/938], Loss: 0.0756
# Epoch [2/4], Step [576/938], Loss: 0.0253
# Epoch [2/4], Step [640/938], Loss: 0.0837
# Epoch [2/4], Step [704/938], Loss: 0.0212
# Epoch [2/4], Step [768/938], Loss: 0.0419
# Epoch [2/4], Step [832/938], Loss: 0.1044
# Epoch [2/4], Step [896/938], Loss: 0.0440
# Epoch [2/4], Step [938/938], Loss: 0.2186
# Loss at epoch 2 0.08613043408274952
# Saving checkpoint......
# Epoch [3/4], Step [64/938], Loss: 0.0613
# Epoch [3/4], Step [128/938], Loss: 0.0403
# Epoch [3/4], Step [192/938], Loss: 0.0224
# Epoch [3/4], Step [256/938], Loss: 0.0414
# Epoch [3/4], Step [320/938], Loss: 0.0439
# Epoch [3/4], Step [384/938], Loss: 0.0263
# Epoch [3/4], Step [448/938], Loss: 0.0678
# Epoch [3/4], Step [512/938], Loss: 0.0124
# Epoch [3/4], Step [576/938], Loss: 0.0621
# Epoch [3/4], Step [640/938], Loss: 0.0137
# Epoch [3/4], Step [704/938], Loss: 0.1644
# Epoch [3/4], Step [768/938], Loss: 0.0114
# Epoch [3/4], Step [832/938], Loss: 0.0085
# Epoch [3/4], Step [896/938], Loss: 0.0597
# Epoch [3/4], Step [938/938], Loss: 0.0047
# Loss at epoch 3 0.059945101094400405
# Epoch [4/4], Step [64/938], Loss: 0.0708
# Epoch [4/4], Step [128/938], Loss: 0.0517
# Epoch [4/4], Step [192/938], Loss: 0.1512
# Epoch [4/4], Step [256/938], Loss: 0.0035
# Epoch [4/4], Step [320/938], Loss: 0.0053
# Epoch [4/4], Step [384/938], Loss: 0.0028
# Epoch [4/4], Step [448/938], Loss: 0.0240
# Epoch [4/4], Step [512/938], Loss: 0.1116
# Epoch [4/4], Step [576/938], Loss: 0.0724
# Epoch [4/4], Step [640/938], Loss: 0.0346
# Epoch [4/4], Step [704/938], Loss: 0.0041
# Epoch [4/4], Step [768/938], Loss: 0.0823
# Epoch [4/4], Step [832/938], Loss: 0.0713
# Epoch [4/4], Step [896/938], Loss: 0.0111
# Epoch [4/4], Step [938/938], Loss: 0.0035
# Loss at epoch 4 0.04647512664595968
# ******** Printing accuracy for Bidirectional LSTM model ***********
# checking accuracy on train data
# Accuracy is 99.09666442871094
# checking accuracy on test data
# Accuracy is 98.85999298095703



import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from custom_data import CustomDataset

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper parameters
in_channel = 3
out_channels = 2
lr = 0.001
batch = 64
epochs = 2


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('Saving checkpoint......')
    torch.save(state, filename)


# load pretrained models
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# load data
dataset = CustomDataset(csv_file='annotations.csv', root_dir='cats_dogs', transform=transforms.ToTensor())
train_data, test_data = random_split(dataset, [1500, 500])
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)
# print(len(train_data))

# loss and optimizer
criterion = nn.CrossEntropyLoss()
op = optim.Adam(model.parameters(), lr=lr)


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
            # print every 4 steps
            if (idx + 1) % 4 == 0:
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
print('******** Printing accuracy for Modified googlenet model on Custom dataset***********')
print('model accuracy for train set')
accuracy_check(train_loader, model)
print('model accuracy for test set')
accuracy_check(test_loader, model)

# ****** training the model********
# Epoch [1/2], Step [4/24], Loss: 2.2950
# Epoch [1/2], Step [8/24], Loss: 0.2212
# Epoch [1/2], Step [12/24], Loss: 0.0652
# Epoch [1/2], Step [16/24], Loss: 0.1858
# Epoch [1/2], Step [20/24], Loss: 0.3523
# Epoch [1/2], Step [24/24], Loss: 0.1157
# Loss at epoch 1 1.10790000964577
# Epoch [2/2], Step [4/24], Loss: 0.0870
# Epoch [2/2], Step [8/24], Loss: 0.0903
# Epoch [2/2], Step [12/24], Loss: 0.0487
# Epoch [2/2], Step [16/24], Loss: 0.1880
# Epoch [2/2], Step [20/24], Loss: 0.0540
# Epoch [2/2], Step [24/24], Loss: 0.0854
# Loss at epoch 2 0.12248363438993692
# ******** Printing accuracy for Modified googlenet model on Custom dataset***********
# model accuracy for train set
# Accuracy is 98.0
# model accuracy for test set
# Accuracy is 92.80000305175781
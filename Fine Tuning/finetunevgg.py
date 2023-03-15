import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models
from torch.utils.data import DataLoader
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper parameters
in_channel = 3
out_channels = 10
lr = 0.001
batch = 512
epochs = 5


class Sample(nn.Module):
    def __init__(self):
        super(Sample, self).__init__()

    def forward(self, x):
        return x


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('Saving checkpoint......')
    torch.save(state, filename)


# load pretrained models
model = torchvision.models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
# Modifying last 2 layers of the model
#   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
#   (classifier): Sequential(
#   (0): Linear(in_features=25088, out_features=4096, bias=True)
#   (1): ReLU(inplace=True)
#   (2): Dropout(p=0.5, inplace=False)
#   (3): Linear(in_features=4096, out_features=4096, bias=True)
#   (4): ReLU(inplace=True)
#   (5): Dropout(p=0.5, inplace=False)
#   (6): Linear(in_features=4096, out_features=1000, bias=True)
model.avgpool = Sample()
model.classifier = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)  # as there are 10 classes in the CIFAR10
)
model.to(device)

# load data
# train_data = datasets.CIFAR10(root='../datasets/', download=True, train=True, transform=transforms.ToTensor())
# train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
# test_data = datasets.CIFAR10(root='../datasets/', download=True, train=False, transform=transforms.ToTensor())
# test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)
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
print('******** Printing accuracy for Modified VGG16 model ***********')
accuracy_check(train_loader, model)
accuracy_check(test_loader, model)

# ****** training the model********
# Epoch [1/5], Step [4/98], Loss: 2.2664
# Epoch [1/5], Step [8/98], Loss: 2.1716
# Epoch [1/5], Step [12/98], Loss: 1.9990
# Epoch [1/5], Step [16/98], Loss: 1.7463
# Epoch [1/5], Step [20/98], Loss: 1.5885
# Epoch [1/5], Step [24/98], Loss: 1.4975
# Epoch [1/5], Step [28/98], Loss: 1.5212
# Epoch [1/5], Step [32/98], Loss: 1.4516
# Epoch [1/5], Step [36/98], Loss: 1.4272
# Epoch [1/5], Step [40/98], Loss: 1.4109
# Epoch [1/5], Step [44/98], Loss: 1.3717
# Epoch [1/5], Step [48/98], Loss: 1.3530
# Epoch [1/5], Step [52/98], Loss: 1.2820
# Epoch [1/5], Step [56/98], Loss: 1.3717
# Epoch [1/5], Step [60/98], Loss: 1.2837
# Epoch [1/5], Step [64/98], Loss: 1.3532
# Epoch [1/5], Step [68/98], Loss: 1.3020
# Epoch [1/5], Step [72/98], Loss: 1.1942
# Epoch [1/5], Step [76/98], Loss: 1.2431
# Epoch [1/5], Step [80/98], Loss: 1.2310
# Epoch [1/5], Step [84/98], Loss: 1.3053
# Epoch [1/5], Step [88/98], Loss: 1.1272
# Epoch [1/5], Step [92/98], Loss: 1.2604
# Epoch [1/5], Step [96/98], Loss: 1.1311
# Epoch [1/5], Step [98/98], Loss: 1.3013
# Loss at epoch 1 1.4449667735975615
# Epoch [2/5], Step [4/98], Loss: 1.2133
# Epoch [2/5], Step [8/98], Loss: 1.1660
# Epoch [2/5], Step [12/98], Loss: 1.1435
# Epoch [2/5], Step [16/98], Loss: 1.1479
# Epoch [2/5], Step [20/98], Loss: 1.1590
# Epoch [2/5], Step [24/98], Loss: 1.1353
# Epoch [2/5], Step [28/98], Loss: 1.2008
# Epoch [2/5], Step [32/98], Loss: 1.1070
# Epoch [2/5], Step [36/98], Loss: 1.1555
# Epoch [2/5], Step [40/98], Loss: 1.1812
# Epoch [2/5], Step [44/98], Loss: 1.1510
# Epoch [2/5], Step [48/98], Loss: 1.1583
# Epoch [2/5], Step [52/98], Loss: 1.1598
# Epoch [2/5], Step [56/98], Loss: 1.0516
# Epoch [2/5], Step [60/98], Loss: 1.1045
# Epoch [2/5], Step [64/98], Loss: 1.1242
# Epoch [2/5], Step [68/98], Loss: 1.0561
# Epoch [2/5], Step [72/98], Loss: 1.1525
# Epoch [2/5], Step [76/98], Loss: 1.1255
# Epoch [2/5], Step [80/98], Loss: 1.1413
# Epoch [2/5], Step [84/98], Loss: 1.1150
# Epoch [2/5], Step [88/98], Loss: 1.0863
# Epoch [2/5], Step [92/98], Loss: 1.1815
# Epoch [2/5], Step [96/98], Loss: 1.1619
# Epoch [2/5], Step [98/98], Loss: 1.1392
# Loss at epoch 2 1.1491739007891442
# Saving checkpoint......
# Epoch [3/5], Step [4/98], Loss: 1.1039
# Epoch [3/5], Step [8/98], Loss: 1.1584
# Epoch [3/5], Step [12/98], Loss: 1.0952
# Epoch [3/5], Step [16/98], Loss: 1.0459
# Epoch [3/5], Step [20/98], Loss: 1.0281
# Epoch [3/5], Step [24/98], Loss: 1.1009
# Epoch [3/5], Step [28/98], Loss: 1.0920
# Epoch [3/5], Step [32/98], Loss: 1.1430
# Epoch [3/5], Step [36/98], Loss: 1.0728
# Epoch [3/5], Step [40/98], Loss: 1.1045
# Epoch [3/5], Step [44/98], Loss: 1.0703
# Epoch [3/5], Step [48/98], Loss: 1.1697
# Epoch [3/5], Step [52/98], Loss: 1.0271
# Epoch [3/5], Step [56/98], Loss: 1.0410
# Epoch [3/5], Step [60/98], Loss: 1.0955
# Epoch [3/5], Step [64/98], Loss: 1.1078
# Epoch [3/5], Step [68/98], Loss: 1.0791
# Epoch [3/5], Step [72/98], Loss: 1.1449
# Epoch [3/5], Step [76/98], Loss: 1.0127
# Epoch [3/5], Step [80/98], Loss: 1.0080
# Epoch [3/5], Step [84/98], Loss: 1.1984
# Epoch [3/5], Step [88/98], Loss: 1.0424
# Epoch [3/5], Step [92/98], Loss: 1.1166
# Epoch [3/5], Step [96/98], Loss: 1.1175
# Epoch [3/5], Step [98/98], Loss: 1.0874
# Loss at epoch 3 1.0887870679096299
# Epoch [4/5], Step [4/98], Loss: 1.0186
# Epoch [4/5], Step [8/98], Loss: 1.0356
# Epoch [4/5], Step [12/98], Loss: 1.0206
# Epoch [4/5], Step [16/98], Loss: 1.0533
# Epoch [4/5], Step [20/98], Loss: 1.0357
# Epoch [4/5], Step [24/98], Loss: 1.0069
# Epoch [4/5], Step [28/98], Loss: 1.0944
# Epoch [4/5], Step [32/98], Loss: 1.0496
# Epoch [4/5], Step [36/98], Loss: 1.0406
# Epoch [4/5], Step [40/98], Loss: 1.0096
# Epoch [4/5], Step [44/98], Loss: 1.0423
# Epoch [4/5], Step [48/98], Loss: 1.0306
# Epoch [4/5], Step [52/98], Loss: 1.0321
# Epoch [4/5], Step [56/98], Loss: 1.1359
# Epoch [4/5], Step [60/98], Loss: 1.0561
# Epoch [4/5], Step [64/98], Loss: 1.1042
# Epoch [4/5], Step [68/98], Loss: 1.0389
# Epoch [4/5], Step [72/98], Loss: 1.0121
# Epoch [4/5], Step [76/98], Loss: 0.9663
# Epoch [4/5], Step [80/98], Loss: 1.1127
# Epoch [4/5], Step [84/98], Loss: 1.1472
# Epoch [4/5], Step [88/98], Loss: 1.0238
# Epoch [4/5], Step [92/98], Loss: 0.9517
# Epoch [4/5], Step [96/98], Loss: 1.0324
# Epoch [4/5], Step [98/98], Loss: 0.9376
# Loss at epoch 4 1.047892770596913
# Saving checkpoint......
# Epoch [5/5], Step [4/98], Loss: 1.0676
# Epoch [5/5], Step [8/98], Loss: 1.0279
# Epoch [5/5], Step [12/98], Loss: 0.9978
# Epoch [5/5], Step [16/98], Loss: 1.0383
# Epoch [5/5], Step [20/98], Loss: 0.9860
# Epoch [5/5], Step [24/98], Loss: 0.9436
# Epoch [5/5], Step [28/98], Loss: 0.9918
# Epoch [5/5], Step [32/98], Loss: 0.9749
# Epoch [5/5], Step [36/98], Loss: 0.9941
# Epoch [5/5], Step [40/98], Loss: 1.0359
# Epoch [5/5], Step [44/98], Loss: 0.9896
# Epoch [5/5], Step [48/98], Loss: 1.0112
# Epoch [5/5], Step [52/98], Loss: 0.9481
# Epoch [5/5], Step [56/98], Loss: 1.0050
# Epoch [5/5], Step [60/98], Loss: 1.1221
# Epoch [5/5], Step [64/98], Loss: 0.9237
# Epoch [5/5], Step [68/98], Loss: 1.0278
# Epoch [5/5], Step [72/98], Loss: 1.0234
# Epoch [5/5], Step [76/98], Loss: 1.0258
# Epoch [5/5], Step [80/98], Loss: 1.0297
# Epoch [5/5], Step [84/98], Loss: 1.0029
# Epoch [5/5], Step [88/98], Loss: 0.9963
# Epoch [5/5], Step [92/98], Loss: 1.0503
# Epoch [5/5], Step [96/98], Loss: 0.9695
# Epoch [5/5], Step [98/98], Loss: 1.0260
# Loss at epoch 5 1.012570852527813
# ******** Printing accuracy for Modified VGG16 model ***********
# checking accuracy on train data
# Accuracy is 65.5979995727539
# checking accuracy on test data
# Accuracy is 62.019996643066406
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

train_data = datasets.CIFAR10(root='../../datasets/', download=True, train=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_data = datasets.CIFAR10(root='../../datasets/', download=True, train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)


class VGG16(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000):
        super(VGG16, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.steps = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.conv_layers = self.create_conv_layers()
        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, self.num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        out = self.fc3(x)

        return out

    def create_conv_layers(self):
        layers = []
        in_channel = self.in_channels
        for each in self.steps:
            if type(each) == int:
                out_channel = each
                layers += [nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)),
                           nn.BatchNorm2d(each),
                           nn.ReLU()
                           ]
                in_channel = each
            elif each == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)


model = VGG16().to(device)


# x = torch.rand((4, 3, 224, 224))
# x = x.to(device)
# print(model(x).shape)


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('Saving checkpoint......')
    torch.save(state, filename)


# loss and criterion
criteria = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# train network
def train_network(num_epochs):
    for e in range(num_epochs):
        losses = []
        if e != 0 and e % 2 == 0:
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            save_checkpoint(checkpoint)
        num_iters = 0
        for idx, (x, y) in enumerate(train_loader):
            num_iters += 1
            start = time.perf_counter()  # start time processing the batch

            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            loss = criteria(scores, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            end = time.perf_counter()  # end time

            if (idx + 1) % batch == 0:
                print(
                    f"Epoch [{e + 1}/{epochs}], "
                    f"Step [{idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}, "
                    f"iteration: {int(num_iters / x.shape[0])}, "
                    f"time: {end - start:.2f}(s), "
                )
            elif (idx + 1) == len(train_loader):
                print(
                    f"Epoch [{e + 1}/{epochs}], "
                    f"Step [{idx + 1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                    f"iteration: {int(num_iters / x.shape[0])}, "
                    f"time: {end - start:.2f}(s), "
                )

        print("Loss at epoch", e + 1, sum(losses) / len(losses))


def check_accuracy(loader, model):
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

            score = model(x)
            _, predictions = score.max(1)  # _ : value corresponding the argmax, predictions = index of max value which the class
            num_correct += (predictions == y).sum()  # increase the number of correct predictions from the batch
            num_samples += predictions.size(0)

    print(f"Accuracy is {(num_correct / num_samples) * 100}")


def main():
    print("****** training the model********")
    train_network(epochs)
    print('******** Printing accuracy for LeNet model ***********')
    # check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)


if __name__ == '__main__':
    main()


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
num_classes = 10
in_channels = 1
out_channels = [6, 16, 120]
lr = 0.001
batch = 64
epochs = 3

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# load dataset
train_data = datasets.MNIST(root='../../datasets/', download=True, train=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)
test_data = datasets.MNIST(root='../../datasets/', download=True, train=False, transform=transform)
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)


class LeNet(nn.Module):
    """
    1. 6 * Convolution ( 5,5), padding = 0, stride =1 and ReLU           torch.Size([64, 6, 28, 28])
    2. AvgPooling kernel (5,5), stride = 2,2                         torch.Size([64, 6, 14, 14])
    3. 16 * Convolution ( 5,5), padding = 0, stride =1 and ReLU           torch.Size([64, 16, 10, 10])
    4. AvgPooling kernel (5,5), stride = 2,2                        torch.Size([64, 16, 5, 5])
    5. 120 * Convolution ( 5,5), padding = 0, stride =1 and ReLU           torch.Size([64, 120, 1, 1])
    6. Linear                                                        torch.Size([64, 84])
    7. Linear                                                        torch.Size([64, 10])
    """
    def __init__(self, in_channels=in_channels, num_classes=num_classes, out_channels = out_channels):
        super(LeNet, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels= 6, kernel_size=(5, 5), padding=(0, 0), stride=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=self.out_channels[0], out_channels=self.out_channels[1], kernel_size=(5, 5), padding=(0, 0), stride=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=self.out_channels[1], out_channels=self.out_channels[2], kernel_size=(5, 5), padding=(0, 0), stride=(1, 1))
        # self.linear = nn.Linear(self.out_channels[1]*5*5, 120)
        self.linear1 = nn.Linear(self.out_channels[2], 84)
        self.linear2 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0], -1)
        # x = self.linear(x)
        x = self.linear1(x)
        out = self.linear2(x)

        return out


model = LeNet().to(device)
# x = torch.rand((64, 1, 32, 32))
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

        for idx, (x, y) in enumerate(train_loader):

            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            loss = criteria(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
            i, predictions = score.max(1)    # i : value corresponding the argmax, predictions = index of max value which the class
            num_correct += (predictions == y).sum()  # increase the number of correct predictions from the batch
            num_samples += predictions.size(0)

    print(f"Accuracy is {(num_correct / num_samples) * 100}")


def main():
    print("****** training the model********")
    train_network(epochs)
    print('******** Printing accuracy for LeNet model ***********')
    check_accuracy(train_loader, model)
    check_accuracy(test_loader, model)


if __name__ == '__main__':
    main()

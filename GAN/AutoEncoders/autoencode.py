import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def train(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    criterion = nn.MSELoss()
    losses = []
    for epoch in range(epochs):
        for idx, (x, y) in enumerate(data):
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = criterion(x, x_hat)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            if (idx + 1) % 64 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{idx + 1}/{len(data)}], "
                    f"Loss: {loss.item():.4f}"
                )
            elif (idx + 1) == len(data):
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Step [{idx + 1}/{len(data)}], "
                    f"Loss: {loss.item():.4f}"
                )
        print("Loss at epoch", epoch+1, sum(losses)/len(losses))
    return autoencoder


latent_dims = 3
autoencoder = Autoencoder(latent_dims).to(device) # GPU

data = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data',
               transform=torchvision.transforms.ToTensor(),
               download=True),
        batch_size=64,
        shuffle=True)

autoencoder = train(autoencoder, data)


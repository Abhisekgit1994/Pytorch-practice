import time
import numpy as np
import torch
import torch.nn as nn
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

base_model = [
    # expansion_ratio, channels, repeat_factor, stride, kernel_size
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [1, 320, 1, 1, 3],

]

phi_values = {
    # alpha: depth, beta:width, gamma:resolution
    # tuple of (phi_value, resolution, drop_rate)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),

}


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size, stride, padding, groups=1):
        """

        :param in_channels: input channels
        :param out_channels: num of out channels for the convolution
        :param kernel_size: kernel size
        :param stride: stride
        :param padding: padding parameter
        :param groups: 1 then it is normal conv, if in_channels then it is depth wise convolution
        """
        super(ConvBlock, self).__init__()
        self.cnn = nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU()

    def forward(self, x):
        return self.silu(self.bn(self.cnn(x)))

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channel, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # C * H * W -> C * 1 * 1
            nn.Conv2d(in_channel, reduced_dim, 1),  # reduce_dim : reduce the number of channels and bring it back
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channel, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = x * self.se(x) # x multiplied by attention scores for each channel
        return out



class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channels, kernel_size, stride, padding, expansion_ratio, reduction=4):
        """

        :param in_channel: input channel
        :param out_channels: out channels
        :param kernel_size: kernel size
        :param stride: stride for conv
        :param padding: padding for conv
        :param expansion_ratio:
        :param reduction: # squeeze excitation
        # survival probability = 0.8 # for stochastic depth
        """
        super(InvertedResidual, self).__init__()
        self.survival_prob = 0.8
        self.use_residual = in_channel==out_channels and stride==1
        hidden_dim = in_channel* expansion_ratio
        self.expand = in_channel != hidden_dim
        reduced_dim = int(in_channel/ reduction)
        if

class EfficientNet(nn.Module):



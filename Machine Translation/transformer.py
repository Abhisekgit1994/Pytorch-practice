import time
import numpy as np
import spacy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pos_encoding import PositionalEmbedding
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import multi30k

spacy_ger = spacy.load('de')
spacy_en = spacy.load('en')
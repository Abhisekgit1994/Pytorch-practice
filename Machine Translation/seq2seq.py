import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter
spacy_germ = spacy.load('de')
spacy_en = spacy.load('en')


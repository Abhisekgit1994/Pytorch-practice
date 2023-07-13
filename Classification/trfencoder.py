import time
import numpy as np
import spacy
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pos_encoding import PositionalEmbedding
from torch.utils.tensorboard import SummaryWriter
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import math
from timeit import default_timer as timer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train, test = AG_NEWS('.', split=('train', 'test'))

tokenizer = get_tokenizer('spacy', 'en_core_web_lg')


def tokenize(data):
    for _, text in data:
        yield tokenizer(text)


PAD, UNK = 0, 1
special_symbols = ['<PAD>', '<UNK>']


vocab = build_vocab_from_iterator(tokenize(train), specials=special_symbols, min_freq=1, special_first=True)
vocab.set_default_index(vocab['<UNK>'])
# torch.save(vocab, 'vocab.pth')
# exit()
print("vocab size", len(vocab))
print(vocab(tokenizer('My name is Abhi and I am passionate about Natural Language Processing')))
# [5997, 988, 25, 1, 10, 311, 3192, 17022, 80, 8011, 11447, 22299]


train = [(text, label-1) for label, text in train]  # originally starts at 1, so
# print(train[0])
# ("Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again.", 2)

test = [(text, label-1) for label, text in test]
# print(test[0])
# ("Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul.", 2)

VOCAB_SIZE = len(vocab)
EMB_SIZE = 512
N_HEADS = 8
FFN_HID_DIM = 512
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 6
NUM_CLASSES = 4
EPOCHS = 10


def tensor_transform(tokens):
    return torch.tensor(tokens)


def sequential_transforms(*transforms):
    def func(text_input):
        for transform in transforms:
            text_input = transform(text_input)
        return text_input
    return func


text_transform = sequential_transforms(tokenizer, vocab, tensor_transform)


def collate_fn(batch):
    x_batch, y_batch =[], []
    for text, label in batch:
        x_batch.append(text_transform(text))
        y_batch.append(label)

    x_batch = pad_sequence(x_batch, padding_value=PAD, batch_first=False)
    y_batch = torch.tensor(y_batch).clone().detach()

    return x_batch, y_batch


train_loader = DataLoader(train, batch_size=BATCH_SIZE, collate_fn=collate_fn)
test_loader = DataLoader(test, batch_size=BATCH_SIZE, collate_fn=collate_fn)


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, dim_ffd, num_classes, dropout=0.1):
        super(Encoder, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dim_ffd = dim_ffd
        self.num_classes = num_classes
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.positional_embed = PositionalEmbedding(max_len=2000, embed_size=self.embed_size)
        assert self.embed_size % self.num_heads == 0, "number of heads must divide evenly into embedding size"

        self.layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=self.num_heads, dim_feedforward=self.dim_ffd, dropout=self.dropout)
        self.encoder = nn.TransformerEncoder(self.layer, num_layers=self.num_layers)

        self.fc = nn.Linear(self.embed_size, self.num_classes)

    def forward(self, x):
        x = self.positional_embed(self.embedding(x))
        x = self.encoder(x)
        x = x.mean(dim=0)
        out = self.fc(x)
        return out


model = Encoder(vocab_size=VOCAB_SIZE, embed_size=EMB_SIZE, num_heads=N_HEADS, num_layers=NUM_ENCODER_LAYERS, dim_ffd=FFN_HID_DIM, num_classes=NUM_CLASSES)
model = model.to(DEVICE)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


def train_epoch(model, optimizer, e):
    print("training started for epoch", e)
    model.train()

    losses = 0
    for idx, (source, target) in enumerate(train_loader):
        source = source.to(DEVICE)
        target = target.to(DEVICE)

        logits = model(source)


        loss = loss_fn(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses += loss.item()
        if (idx + 1) % BATCH_SIZE == 0:
            print(
                f"Epoch [{e}/{EPOCHS}], "
                f"Step [{idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )
        elif (idx + 1) == len(train_loader):
            print(
                f"Epoch [{e}/{EPOCHS}], "
                f"Step [{idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )
    return losses / len(list(train_loader))


# running_loss = float('inf')
for epoch in range(1, EPOCHS + 1):
    start_time = timer()
    train_loss = train_epoch(model, optimizer, epoch)
    end_time = timer()
    # val_loss = evaluate(model)
    print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'my_model.pth.tar')


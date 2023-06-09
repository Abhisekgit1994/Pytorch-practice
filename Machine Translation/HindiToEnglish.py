# Reference:
# https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus
# https://huggingface.co/datasets/cfilt/iitb-english-hindi
# https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
import joblib
import math
import pickle
import time
from timeit import default_timer as timer
import deepspeed
import numpy as np
import pandas as pd
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from inltk.inltk import setup
from inltk.inltk import tokenize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from pos_encoding import PositionalEmbedding
setup('hi')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
print(DEVICE)

SOURCE_LANG = 'hi'
TARGET_LANG = 'en'

PAD, UNK, SOS, EOS = 0, 1, 2, 3
special_symbols = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

# convert text to tokens
token_transform = {SOURCE_LANG: tokenize, TARGET_LANG: get_tokenizer('spacy', language='en_core_web_sm')}


def yield_tokens(data_iter, lang):
    language_index = {SOURCE_LANG: 0, TARGET_LANG: 1}
    for i, data in enumerate(data_iter):
        if lang == SOURCE_LANG:
            temp = token_transform[lang](data[language_index[lang]], lang)
            temp = [x.replace('▁', '') for x in temp if '▁' in x]  # to replace space in hindi strings
            yield temp
        else:
            yield token_transform[lang](data[language_index[lang]])


with open('D:/Abhi/COURSERA/Machine Translation data/Hindi To English/vocab/vocab.pkl', 'rb') as file:
    vocab_transform = pickle.load(file)


def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS])
                      ))


def sequential_transforms(*transforms):
    def func(text):
        for transform in transforms:
            if transform == tokenize:
                text = transform(text, 'hi')
            else:
                text = transform(text)
        return text

    return func


text_transform = {}
for lang in [SOURCE_LANG, TARGET_LANG]:
    text_transform[lang] = sequential_transforms(token_transform[lang], vocab_transform[lang], tensor_transform)


def collate_fn(batch):
    source_batch, target_batch = [], []
    for source, target in batch:
        source_batch.append(text_transform[SOURCE_LANG](source.replace("▁", " ").rstrip("\n")))
        target_batch.append(text_transform[TARGET_LANG](target.rstrip("\n")))

    source_batch = pad_sequence(source_batch, padding_value=PAD, batch_first=False)
    target_batch = pad_sequence(target_batch, padding_value=PAD, batch_first=False)

    return source_batch, target_batch


SOURCE_VOCAB_SIZE = len(vocab_transform[SOURCE_LANG])
TARGET_VOCAB_SIZE = len(vocab_transform[TARGET_LANG])
EMBED_SIZE = 512
N_HEADS = 8
FFN_HID_DIM = 512
BATCH_SIZE = 8
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

print(TARGET_VOCAB_SIZE)

with open('D:/Abhi/COURSERA/Machine Translation data/Hindi To English/data/train_hindiToEnglish.pkl', 'rb') as file:
    train_iter = pd.read_pickle(file)[:500000]
train_iter = [each for each in train_iter if len(each[0].split()) < 10]
print(len(train_iter))
train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
with open('D:/Abhi/COURSERA/Machine Translation data/Hindi To English/data/val_hindiToEnglish.pkl', 'rb') as file:
    val_iter = pd.read_pickle(file)
val_loader = DataLoader(val_iter, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
print(len(val_loader))


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, tokens):
        return self.embedding(torch.tensor(tokens))


def generate_square_subsequent_mask(size):
    mask = (torch.tril(torch.ones((size, size), device=DEVICE)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def createMask(source, target):
    source_len = source.shape[0]
    target_len = target.shape[0]

    target_mask = generate_square_subsequent_mask(target_len)
    source_mask = torch.zeros((source_len, source_len), device=DEVICE).type(torch.bool)

    source_padding_mask = (source == PAD).transpose(0, 1)
    target_padding_mask = (target == PAD).transpose(0, 1)

    return source_mask, target_mask, source_padding_mask, target_padding_mask


class HindiToEngTransformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embed_size, num_head, num_enc_layers, num_dec_layers, dim_feedforward, dropout=0.1):
        """

        :param source_vocab_size: vocab size for input
        :param target_vocab_size: vocab size for output
        :param embed_size: the number of expected features in the encoder/decoder inputs (default=512)
        :param num_head: the number of heads in the multi head attention models (default=8).
        :param num_enc_layers: the number of sub-encoder-layers in the encoder (default=6).
        :param num_dec_layers: the number of sub-decoder-layers in the decoder (default=6).
        :param dim_feedforward: the dimension of the feedforward network model (default=2048).
        :param dropout: dropout value
        """
        super(HindiToEngTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_head, num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.source_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.target_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_embedding = PositionalEmbedding(max_len=5000, embed_size=embed_size)

    def forward(self, source, target, source_mask, target_mask, source_padding_mask, target_padding_mask, memory_key_padding_mask):
        """
        :param source: the sequence to the encoder (required).
        :param target: the sequence to the decoder (required).
        :param source_mask: the additive mask for the src sequence (optional).
        :param target_mask: the additive mask for the tgt sequence (optional).
        :param source_padding_mask: the Tensor mask for src keys per batch (optional)
        :param target_padding_mask: the Tensor mask for tgt keys per batch (optional).
        :param memory_key_padding_mask: the Tensor mask for memory keys per batch
        :return:
        """
        x = self.positional_embedding(self.source_embedding(source))
        y = self.positional_embedding(self.target_embedding(target))

        out = self.transformer(x, y, source_mask, target_mask, None, source_padding_mask, target_padding_mask, memory_key_padding_mask)

        out = self.fc_out(out)

        return out

    def encode(self, source, source_mask):
        self.transformer.encoder(self.positional_embedding(self.source_embedding(source)), source_mask)

    def decode(self, target, memory, target_mask):
        self.transformer.decoder(self.positional_embedding(self.target_embedding(target)), memory, target_mask)


# source_vocab_size, target_vocab_size, embed_size, num_head, num_enc_layers, num_dec_layers, dim_feedforward, dropout=0.1
model = HindiToEngTransformer(SOURCE_VOCAB_SIZE, TARGET_VOCAB_SIZE, EMBED_SIZE, N_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FFN_HID_DIM)
model = model.to(DEVICE)
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-9)


def load_checkpoint(checkpoint):
    print("Loading checkpoint.....")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


load_checkpoint(torch.load('../../Machine Translation data/Hindi To English/my_model.pth.tar'))


def train_model(model, optimizer, epoch):
    model.train()

    losses = 0
    for idx, (source, target) in enumerate(train_loader):
        source = source.to(DEVICE)
        target = target.to(DEVICE)
        target_in = target[:-1, :]
        source_mask, target_mask, source_padding_mask, target_padding_mask = createMask(source, target_in)
        logits = model(source, target_in, source_mask, target_mask, source_padding_mask, target_padding_mask, source_padding_mask)

        target_out = target[1:, :]

        optimizer.zero_grad()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        loss.backward()
        optimizer.step()

        losses += loss.item()

        if (idx + 1) % 8 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}], "
                f"Step [{idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )
        elif (idx + 1) == len(train_loader):
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}], "
                f"Step [{idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )
        if (idx + 1) % 256 == 0:
            print("Saving checkpoint during the epoch.......")
            checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, '../../Machine Translation data/Hindi To English/my_model.pth.tar')

    return losses / len(list(train_loader))


def evaluate(model,epoch):
    print("Running validation.....")
    model.eval()
    losses = 0
    for idx, (source, target) in enumerate(val_loader):
        source = source.to(DEVICE)
        target = target.to(DEVICE)
        target_in = target[:-1, :]
        source_mask, target_mask, source_padding_mask, target_padding_mask = createMask(source, target_in)
        logits = model(source, target_in, source_mask, target_mask, source_padding_mask, target_padding_mask, source_padding_mask)
        target_out = target[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        losses += loss.item()
        if (idx + 1) % 8 == 0:
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}], "
                f"Step [{idx + 1}/{len(val_loader)}], "
                f"Loss: {loss.item():.4f}"
            )
        elif (idx + 1) == len(train_loader):
            print(
                f"Epoch [{epoch}/{NUM_EPOCHS}], "
                f"Step [{idx + 1}/{len(val_loader)}], "
                f"Loss: {loss.item():.4f}"
            )

    return losses / len(list(val_loader))


NUM_EPOCHS = 8
running_loss = float('inf')
for epoch in range(1, NUM_EPOCHS + 1):
    start_time = timer()
    train_loss = train_model(model, optimizer, epoch)
    end_time = timer()
    val_loss = evaluate(model,epoch)
    print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
    if val_loss < running_loss:
        print(f"running loss: {running_loss}, Val loss: {val_loss:.3f}")
        print("Saving checkpoint.......")
        checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, '../../Machine Translation data/Hindi To English/my_model.pth.tar')
        running_loss = val_loss

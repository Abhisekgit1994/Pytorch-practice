# Reference: https://pytorch.org/tutorials/beginner/translation_transformer.html

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
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import math
from timeit import default_timer as timer

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SOURCE_LANG = 'de'
TARGET_LANG = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}

token_transform[SOURCE_LANG] = get_tokenizer('spacy', language='de_core_web_sm')
token_transform[TARGET_LANG] = get_tokenizer('spacy', language='en_core_web_sm')


# print(token_transform)


def yield_tokens(data_iter, language):
    language_index = {SOURCE_LANG: 0, TARGET_LANG: 1}

    for data in data_iter:
        yield token_transform[language](data[language_index[language]])


PAD, UNK, SOS, EOS = 0, 1, 2, 3

special_symbols = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
for lang in [SOURCE_LANG, TARGET_LANG]:
    # train data iterator
    train_iter = Multi30k(split='train', language_pair=(SOURCE_LANG, TARGET_LANG))
    # create torch text's vocab object
    vocab_transform[lang] = build_vocab_from_iterator(yield_tokens(train_iter, lang), min_freq=1, specials=special_symbols, special_first=True)

for lang in [SOURCE_LANG, TARGET_LANG]:
    vocab_transform[lang].set_default_index(UNK)

print(vocab_transform)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(TokenEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, tokens):
        return self.embedding(torch.tensor(tokens))


class TranslationTransformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embed_size, num_heads, num_enc_layers, num_dec_layers, dim_feedforward, dropout=0.1):
        super(TranslationTransformer, self).__init__()

        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_enc_layers, num_decoder_layers=num_dec_layers, dim_feedforward=dim_feedforward, dropout=dropout)
        self.output = nn.Linear(embed_size, target_vocab_size)  # use to convert embed size to number of vocab in the language

        self.source_embedding = nn.Embedding(source_vocab_size, embed_size)
        self.target_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_embedding = PositionalEmbedding(max_len=5000, embed_size=embed_size)

    def forward(self, source, target, source_mask, target_mask, source_padding, target_padding, memory_key_padding_mask):
        x = self.positional_embedding(self.source_embedding(source))
        y = self.positional_embedding(self.target_embedding(target))

        # forward(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)
        out = self.transformer(x, y, source_mask, target_mask, None, source_padding, target_padding, memory_key_padding_mask)

        out = self.output(out)
        return out

    def encode(self, source, source_mask):
        # print(help(self.transformer.encoder))
        return self.transformer.encoder(self.positional_embedding(self.source_embedding(source)), source_mask)

    def decode(self, target, memory, target_mask):
        return self.transformer.decoder(self.positional_embedding(self.target_embedding(target)), memory, target_mask)


def generate_square_subsequent_mask(size):
    mask = (torch.tril(torch.ones((size, size), device=DEVICE)) == 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(source, target):
    source_len = source.shape[0]
    target_len = target.shape[0]

    target_mask = generate_square_subsequent_mask(target_len)
    source_mask = torch.zeros((source_len, source_len), device=DEVICE).type(torch.bool)

    source_padding_mask = (source == PAD).transpose(0, 1)
    target_padding_mask = (target == PAD).transpose(0, 1)

    return source_mask, target_mask, source_padding_mask, target_padding_mask


SRC_VOCAB_SIZE = len(vocab_transform[SOURCE_LANG])
TGT_VOCAB_SIZE = len(vocab_transform[TARGET_LANG])
EMB_SIZE = 512
N_HEADS = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

print(SRC_VOCAB_SIZE)

# source_vocab_size, target_vocab_size, embed_size, num_heads, num_enc_layers, num_dec_layers, dim_feedforward, dropout=0.1
transformer = TranslationTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE, N_HEADS, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, FFN_HID_DIM)

transformer = transformer.to(device=DEVICE)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


# COLLATION
def tensor_transform(token_ids):
    return torch.cat((torch.tensor([SOS]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS])
                      ))


def sequential_transforms(*transforms):
    def func(text_input):
        for transform in transforms:
            text_input = transform(text_input)
        return text_input

    return func


text_transform = {}
for lang in [SOURCE_LANG, TARGET_LANG]:
    text_transform[lang] = sequential_transforms(token_transform[lang], vocab_transform[lang], tensor_transform)

# print(text_transform[SOURCE_LANG])


def collate_fn(batch):
    source_batch, target_batch = [], []
    for source, target in batch:
        source_batch.append(text_transform[SOURCE_LANG](source.rstrip("\n")))
        target_batch.append(text_transform[TARGET_LANG](target.rstrip("\n")))

    source_batch = pad_sequence(source_batch, padding_value=PAD, batch_first=False)
    target_batch = pad_sequence(target_batch, padding_value=PAD, batch_first=False)

    return source_batch, target_batch


train_iter = Multi30k(split='train', language_pair=(SOURCE_LANG, TARGET_LANG))
train_loader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
valid_iter = Multi30k(split='valid', language_pair=(SOURCE_LANG, TARGET_LANG))
valid_loader = DataLoader(valid_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)


def train_epoch(model, optimizer):
    model.train()

    losses = 0
    for source, target in train_loader:
        source = source.to(DEVICE)
        target = target.to(DEVICE)

        target_in = target[:-1, :]  # not taking the last column so that we will predict that using our model
        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source, target_in)
        logits = model(source, target_in, source_mask, target_mask, source_padding_mask, target_padding_mask, source_padding_mask)

        optimizer.zero_grad()
        target_out = target[1:, :]
        # print(target_out.shape)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        loss.backward()
        optimizer.step()

        losses += loss.item()

    return losses / len(list(train_loader))


def evaluate(model):
    model.eval()
    losses = 0

    for source, target in valid_loader:
        source = source.to(DEVICE)
        target = target.to(DEVICE)

        target_in = target[:-1, :]
        source_mask, target_mask, source_padding_mask, target_padding_mask = create_mask(source, target_in)
        logits = model(source, target_in, source_mask, target_mask, source_padding_mask, target_padding_mask, source_padding_mask)
        target_out = target[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), target_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(valid_loader))


load_model = False


def load_checkpoint(checkpoint):
    print("Loading checkpoint.....")
    transformer.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


if not load_model:
    NUM_EPOCHS = 15
    running_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer)
        end_time = timer()
        val_loss = evaluate(transformer)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
        if val_loss < running_loss:
            print(f"running loss: {running_loss}, Val loss: {val_loss:.3f}")
            print("Saving checkpoint.......")
            checkpoint = {'state_dict': transformer.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, 'my_model.pth.tar')
            running_loss = val_loss
else:
    load_checkpoint(torch.load('my_model.pth.tar'))


def greedy_decode(model, source, source_mask, max_len, start_symbol):
    source = source.to(DEVICE)
    source_mask = source_mask.to(DEVICE)

    memory = transformer.encode(source, source_mask)

    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)

        target_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool).to(DEVICE)

        out = model.decode(ys, memory, target_mask)
        out = out.transpose(0, 1)
        probs = model.output(out[:, -1])
        _, next_word = torch.max(probs, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(source.data).fill_(next_word)], dim=0)
        if next_word == EOS:
            break
    return ys


# function to translate input to target
def translate(model, source_sent):
    model.eval()
    source = text_transform[SOURCE_LANG](source_sent).view(-1, 1)

    num_tokens = source.shape[0]

    source_mask = torch.zeros((num_tokens, num_tokens)).type(torch.bool)

    target_tokens = greedy_decode(model, source, source_mask, max_len=num_tokens + 5, start_symbol=SOS).flatten()

    return " ".join(vocab_transform[TARGET_LANG].lookup_tokens(list(target_tokens.cpu().numpy()))).replace("<SOS>", "").replace("<EOS>", "")


if __name__ == '__main__':
    print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))


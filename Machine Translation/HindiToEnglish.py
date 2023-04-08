# Reference:
# https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus
# https://huggingface.co/datasets/cfilt/iitb-english-hindi
import pickle
import time
import numpy as np
import pandas as pd
from inltk.inltk import tokenize
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
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
import math, joblib
from timeit import default_timer as timer
from inltk.inltk import setup
from pos_encoding import PositionalEmbedding
# from transformer import TokenEmbedding
# from transformer import TranslationTransformer, tensor_transform, sequential_transforms, create_mask, generate_square_subsequent_mask

setup('hi')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

SOURCE_LANG = 'hi'
TARGET_LANG = 'en'

PAD, UNK, SOS, EOS = 0, 1, 2, 3
special_symbols = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']

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


# Create vocab for first 150k rows of the data and save them in a pkl file so that we can directly use the vocab later without creating it.
"""
with open('train_hindiToEnglish.pkl', 'rb') as file:
    train_iter = pd.read_pickle(file)[:150000]
with open('val_hindiToEnglish.pkl', 'rb') as file:
    val_iter = pd.read_pickle(file)
for lang in [SOURCE_LANG, TARGET_LANG]:
    vocab_transform[lang] = build_vocab_from_iterator(yield_tokens(train_iter, lang), min_freq=3, specials=special_symbols, special_first=True, )
for lang in [SOURCE_LANG, TARGET_LANG]:
    vocab_transform[lang].set_default_index(UNK)
# saving the vocabs as pickle to save creation time
with open('vocab.pkl', 'wb') as file:
    pickle.dump(vocab_transform, file)
    
"""
with open('D:/Abhi/COURSERA/Machine Transalation Models/Hindi To English/vocab/vocab.pkl', 'rb') as file:
    vocab_transform = pickle.load(file)

# print(vocab_transform['hi'].get_stoi())
# print(vocab_transform['en'].get_stoi())
# text_transform = {}
# for lang in [SOURCE_LANG, TARGET_LANG]:
#     text_transform[lang] = sequential_transforms(token_transform[lang], vocab_transform[lang], tensor_transform)
#
#
# def collate_fn(batch):
#     source_batch, target_batch = [], []
#     for source, target in batch:
#         source_batch.append(text_transform[SOURCE_LANG](source.replace("▁", " ").rstrip("\n")))
#         target_batch.append(text_transform[TARGET_LANG](target.rstrip("\n")))
#
#     source_batch = pad_sequence(source_batch, padding_value=PAD, batch_first=False)
#     target_batch = pad_sequence(target_batch, padding_value=PAD, batch_first=False)
#
#     return source_batch, target_batch


SRC_VOCAB_SIZE = len(vocab_transform[SOURCE_LANG])
TGT_VOCAB_SIZE = len(vocab_transform[TARGET_LANG])
EMB_SIZE = 512
N_HEADS = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 6
NUM_DECODER_LAYERS = 6

print(TGT_VOCAB_SIZE)



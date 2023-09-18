import csv
import os
from collections import Counter
from datetime import datetime
from math import ceil
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.nn.init import xavier_uniform_
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torchtext.vocab import GloVe, build_vocab_from_iterator, vocab, Vocab
from tqdm.notebook import tqdm
from torchcrf import CRF

mask = {'<start>': 1, 's-org': 1, 'o': 1, 's-misc': 1, '<pad>': 0,
        '<end>': 1, 'b-per': 1, 'e-per': 1, 's-loc': 1, 'b-org': 1,
        'e-org': 1, 'i-per': 1, 's-per': 1, 'b-misc': 1, 'i-misc': 1,
        'e-misc': 1, 'i-org': 1, 'b-loc': 1, 'e-loc': 1, 'i-loc': 1}

sentence_length = 30
batch_sizes = [16, 16, 16]
dropout = 0.2
embedding_dim = 300  # using GloVe with 100 dimensions
resume_training = False
hidden_size = 256
optimizers = ['adam']
learning_rates = [(6e-4, 1.2e-3), (4e-4, 9e-4), (4e-4, 9e-4)]
weight_decays = [1e-4]
max_epochs = [200, 200, 200]
grad_norm_clipping = 2
gamma = 0.99998
PAD = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'running at {device}')


class ConLLData(Dataset):
    def __init__(self, df, max_len, class_mask, stop_words=None, special_tags=None, min_freq=1):
        super(ConLLData, self).__init__()
        self.special_tags = ['<pad>', '<start>', '<end>'] if special_tags is None else special_tags
        self.class_mask = class_mask
        self.sentences, self.sentence_masks, self.pos, self.chunks, self.entities, self.documents = self.preprocess(df, max_len, stop_words)

        self.vocab = build_vocab_from_iterator(self.yield_tokens(self.documents), min_freq=1, specials=self.special_tags)
        # print(self.sentence_masks)
        # print(self.vocab.get_stoi())
        # self.pos_tags = vocab(Counter([str(t) for s in self.pos for t in s]), min_freq=1, specials=self.special_tags)
        # self.chunk_tags = vocab(Counter([str(t) for s in self.chunks for t in s]), min_freq=1, specials=self.special_tags)
        self.entity_tags = build_vocab_from_iterator(self.yield_tokens(self.entities), min_freq=1, specials=self.special_tags, special_first=True)

        self.input = [self.vocab(s) for s in self.sentences]
        print(self.input[0])
        self.target = [self.entity_tags(s) for s in self.entities]
        print(self.target[0])

    def yield_tokens(self, documents):
        for each in documents:
            yield each

    def padsequences(self, sentence, length, pad_tag='<pad>'):
        sentence = [self.special_tags[1]] + sentence + [self.special_tags[2]]
        return sentence

    def preprocess(self, df, max_len, stop_words=None):
        sentences = []
        pos_tags = []
        chunk_tags = []
        entities = []
        masks = []
        document_sentences = []

        sentence_end_indices = df[df['token'].isnull()].index.tolist()
        for i in range(len(sentence_end_indices)-1):
            start = sentence_end_indices[i]
            end = sentence_end_indices[i+1]
            curr_len = end-start

            for j in range(start, end, max_len):
                if df.token.iloc[j] == '-DOCSTART-':
                    j += 1
                else:
                    k = j + curr_len

                sent = df.iloc[j:k].dropna(how='any', axis='rows').replace('\d+', '0', regex=True) # replace any number with 0

                if stop_words is not None:
                    idx = sent.word[sent.word.isin(stop_words)].index.to_list()
                    sent = sent.drop(idx)

                if len(sent) > 1:
                    sentences.append(self.padsequences(sent['token'].to_list(), max_len))
                    pos_tags.append(self.padsequences(sent['pos'].str.lower().to_list(), max_len))
                    chunk_tags.append(self.padsequences(sent['chunk'].str.lower().to_list(), max_len))
                    entities.append(self.padsequences(sent['entity'].str.lower().to_list(), max_len))
                    masks.append([self.class_mask[t] for t in entities[-1]])
                    document_sentences.append(sent['token'].to_list())
        return sentences, masks, pos_tags, chunk_tags, entities, document_sentences

    def __getitem__(self, index):
        return self.input[index], self.sentence_masks[index], self.target[index], self.sentences[index]

    def __len__(self):
        return len(self.sentences)


df = pd.read_csv('conll data/train.txt', sep=' ', skip_blank_lines=False, names=['token', 'pos', 'chunk', 'entity'])
train = ConLLData(df, sentence_length, mask, min_freq=1)
# print(train.vocab.get_stoi())


def collate_fn(batch):
    inputs, masks, target, texts = [], [], [], []
    for x, mask, y, text in batch:
        inputs.append(torch.tensor(x))
        masks.append(torch.tensor(mask))
        target.append(torch.tensor(y))
        texts.append(text)
    inputs = pad_sequence(inputs, padding_value=train.vocab.get_stoi()['<pad>'], batch_first=True)
    masks = pad_sequence(masks, padding_value=train.entity_tags.get_stoi()['<pad>'], batch_first=True)
    target = pad_sequence(target, padding_value=train.entity_tags.get_stoi()['<pad>'], batch_first=True)

    return inputs, masks, target, texts


train_loader = DataLoader(train, batch_size=16, collate_fn=collate_fn)


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, entity_tags, embed_size, hidden_dim, num_layers, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.tag_to_ix = entity_tags
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.target_size = len(self.tag_to_ix)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_dim, num_layers=self.num_layers, bidirectional=True, batch_first=True)

        # map the output of the LSTM to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim * 2, self.target_size)
        self.crf = CRF(self.target_size, batch_first=True)

    def _get_lstm_features(self, sentence):
        h0, c0 = xavier_uniform_(torch.zeros(self.num_layers * 2, sentence.size(0), self.hidden_dim)).to(device), xavier_uniform_(torch.zeros(self.num_layers * 2, sentence.size(0), self.hidden_dim)).to(device)
        embeds = self.embedding(sentence)
        out, (hn, cn) = self.lstm(embeds, (h0, c0))
        lstm_feats = self.hidden2tag(out)
        return lstm_feats

    def neg_log_likelihood(self, sentence, tags, mask):
        feats = self._get_lstm_features(sentence)
        llh = self.crf(feats, tags, mask, reduction='token_mean')
        return -llh

    def forward(self, sentence):  # don't confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)
        # Find the best path, given the features.
        # score, path = self._viterbi_decode(lstm_feats)
        # score1, path1 = self._viterbi_decode_new(lstm_feats)
        return self.crf.decode(lstm_feats)


model = BiLSTM_CRF(len(train.vocab.get_stoi()), train.entity_tags.get_stoi(), embedding_dim, hidden_size, 3, 64).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def training_epoch(model, optimizer, train_loader):  # criterion, scheduler
    model.train()
    losses = 0
    c = 0
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for x, mask, y, text in progress_bar:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        mask = mask.bool() # Convert mask tensor to boolean otherwise it does not work with crf layers

        nll = model.neg_log_likelihood(x, y, mask)
        model.zero_grad()
        nll.backward()
        optimizer.step()
        losses += nll.item()
        c += 1
        if (c+1) % 64 == 0:
            print(
                f"Step [{c + 1}/{len(train_loader)}], "
                f"Loss: {nll.item():.4f}"
            )
        elif (c + 1) == len(train_loader):
            print(
                f"Step [{c + 1}/{len(train_loader)}], "
                f"Loss: {nll.item():.4f}"
            )

    return losses / len(list(train_loader))


for i in range(100):
    start_time = timer()
    loss = training_epoch(model, optimizer, train_loader)
    end_time = timer()
    print(f"Epoch: {i+1}, Train loss: {loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s")
    checkpoint = {'epoch': i, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'loss':loss}
    torch.save(checkpoint, 'conll.pt')


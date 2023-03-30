import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pos_encoding import createPositionalEncoding


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embedding_size
        self.num_heads = num_heads
        self.head_dims = self.embedding_size // self.num_heads

        assert (self.num_heads * self.head_dims == self.embed_size), "Embedding size should be divisible y number of heads"

        # Linear layer for query
        self.query_layer = nn.Linear(self.embed_size, self.embed_size, bias=False)  # If set to False, the layer will not learn an additive bias. Default: True
        # Linear layer for Key
        self.key_layer = nn.Linear(self.embed_size, self.embed_size, bias=False)
        # Linear layer for Value
        self.value_layer = nn.Linear(self.embed_size, self.embed_size, bias=False)

        self.fc_out = nn.Linear(self.embed_size, self.embed_size)

    def forward(self, queries, keys, values, mask):
        N = queries.shape[0]
        len_val, len_key, len_que = values.shape[1], keys.shape[1], queries.shape[1]  # seq_len for each matrix

        queries = self.query_layer(queries)  # (N * len_que * embed_size)
        values = self.value_layer(values)  # (N * len_val * embed_size
        keys = self.key_layer(keys)  # (N * len_key * embed_size)

        # adding explicit head dimensions : split last embed_size into num_heads * head_dims
        values = values.reshape(N, len_val, self.num_heads, self.head_dims)
        queries = queries.reshape(N, len_que, self.num_heads, self.head_dims)
        keys = keys.reshape(N, len_key, self.num_heads, self.head_dims)

        first_step = torch.einsum("nhqk", [queries, keys])  # (N * num_heads, len_que, len_key)

        if mask is not None:
            first_step = first_step.masked_fill(mask==0, float("-1e20"))

        attention = torch.softmax(first_step / self.embed_size ** (1 / 2), dim=3)

        out = torch.einsum([attention, values]).reshape(N, len_que, self.num_heads* self.head_dims)

        out = self.fc_out(out) # (N * len_que, embed_size)

        return out










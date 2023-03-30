import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from pos_encoding import PositionalEmbedding


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        """
        Multi head attention
        :param embedding_size: input embedding size
        :param num_heads: number of split heads
        """
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
        """
        Suppose we have batch_size=32,sequence_length=10, embedding dimension=512. So after embedding and positional encoding our output will be of dimension 32x10x512.
        We will resize it to 32x10x8x64.(About 8, it is the number of heads in multi head attention) and then bring it back to 32* 10 * 512
        """
        N = queries.shape[0]
        len_val, len_key, len_que = values.shape[1], keys.shape[1], queries.shape[1]  # seq_len for each matrix

        queries = self.query_layer(queries)  # (N * len_que * embed_size)
        values = self.value_layer(values)  # (N * len_val * embed_size
        keys = self.key_layer(keys)  # (N * len_key * embed_size)

        # adding explicit head dimensions : split last embed_size into num_heads * head_dims
        values = values.reshape(N, len_val, self.num_heads, self.head_dims)  # 32 * 10 * 8 * 64
        queries = queries.reshape(N, len_que, self.num_heads, self.head_dims)
        keys = keys.reshape(N, len_key, self.num_heads, self.head_dims)

        first_step = torch.einsum("nhqk", [queries, keys])  # (N * num_heads, len_que, len_key)

        if mask is not None:
            first_step = first_step.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(first_step / self.embed_size ** (1 / 2), dim=3)
        # Collapse the Head dimension by reshaping to (Batch, Sequence, Head * Query size). This effectively concatenates the Attention Score vectors for each head into a single merged Attention Score.
        out = torch.einsum([attention, values]).reshape(N, len_que, self.num_heads * self.head_dims) # 32 * 10 * 512

        out = self.fc_out(out)  # (N * len_que, embed_size)

        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout, expansion_factor):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size=embed_size, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*expansion_factor),
            nn.ReLU(),
            nn.Linear(embed_size*expansion_factor, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask):
        attention = self.attention(query, key, value, mask)
        x = self.dropout(self.norm1(attention + query))  # skip connection for 1st step to the layer norm
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2([forward + x]))  # 2nd skip connection to the 2nd layer norm
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, device, expansion_factor, dropout, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.max_length = max_length
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.positional_embedding = PositionalEmbedding(seq_len=max_length, embed_size=self.embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size=self.embed_size, num_heads=self.num_heads, dropout=dropout, expansion_factor=self.expansion_factor)

        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        pass









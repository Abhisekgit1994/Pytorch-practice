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
        self.head_dims = self.embed_size // self.num_heads

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
        # print(queries.shape)
        # print(keys.shape)
        N = queries.shape[0]
        len_val, len_key, len_que = values.shape[1], keys.shape[1], queries.shape[1]  # seq_len for each matrix

        queries = self.query_layer(queries)  # (N * len_que * embed_size)
        values = self.value_layer(values)  # (N * len_val * embed_size
        keys = self.key_layer(keys)  # (N * len_key * embed_size)

        # adding explicit head dimensions : split last embed_size into num_heads * head_dims
        values = values.reshape(N, len_val, self.num_heads, self.head_dims)  # 32 * 10 * 8 * 64
        queries = queries.reshape(N, len_que, self.num_heads, self.head_dims)
        keys = keys.reshape(N, len_key, self.num_heads, self.head_dims)

        first_step = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N * num_heads, len_que, len_key)

        if mask is not None:
            first_step = first_step.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(first_step / self.embed_size ** (1 / 2), dim=3)
        # Collapse the Head dimension by reshaping to (Batch, Sequence, Head * Query size). This effectively concatenates the Attention Score vectors for each head into a single merged Attention Score.
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, len_que, self.num_heads * self.head_dims) # 32 * 10 * 512

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
        out = self.dropout(self.norm2(forward + x))  # 2nd skip connection to the 2nd layer norm
        return out


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, device, expansion_factor, dropout, seq_len, max_length):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.expansion_factor = expansion_factor
        self.max_length = max_length
        self.seq_len = seq_len
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.positional_embedding = PositionalEmbedding(max_len=self.max_length, embed_size=self.embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size=self.embed_size, num_heads=self.num_heads, dropout=dropout, expansion_factor=self.expansion_factor)
            for i in range(self.num_layers)

        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_len = x.shape
        x = self.word_embedding(x)
        # positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        # print(positions.shape)
        out = self.positional_embedding(x)
        for layer in self.layers:
            out = layer(out, out, out, mask)  # input to the transformer block query, key, and value is out : input embedding + positional encoding

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, expansion_factor, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, num_heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(embed_size, num_heads, dropout, expansion_factor)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, source_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(self.norm(attention+x))
        out = self.transformer_block(query, key, value, source_mask)

        return out


class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, num_layers, num_heads, expansion_factor, dropout, device, seq_len, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.seq_len = seq_len
        # self.seq_len = 7
        self.word_embedding = nn.Embedding(target_vocab_size, embed_size)
        self.positional_encoding = PositionalEmbedding(self.seq_len, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, num_heads, expansion_factor, dropout, device) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, source_mask, target_mask):
        N, seq_len = x.shape
        x = self.word_embedding(x)
        x = self.dropout(self.positional_encoding(x))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, source_mask, target_mask)

        out = self.fc_out(x)
        return out


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_pad_idx, target_pad_idx, seq_len, embed_size=512, num_layers=6, expansion_factor=4, num_heads=8, dropout=0, device='cpu', max_length=100):
        super(Transformer, self).__init__()

        # source vocab_size, embed_size, num_layers, num_heads, device, expansion_factor, dropout, max_length
        self.seq_len = seq_len
        self.encoder = Encoder(source_vocab_size, embed_size, num_layers, num_heads, device, expansion_factor, dropout, seq_len=self.seq_len, max_length=max_length)
        # target_vocab_size, embed_size, num_layers, num_heads, expansion_factor, dropout, device, max_length
        self.decoder = Decoder(target_vocab_size, embed_size, num_layers, num_heads, expansion_factor, dropout, device, seq_len=self.seq_len, max_length=max_length)

        self.source_pad_idx = source_pad_idx
        self.target_pad_idx = target_pad_idx

        self.device = device

    def make_source_mask(self, source):
        source_mask = (source != self.source_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N , 1, 1, source_len)
        return source_mask.to(self.device)

    def make_target_mask(self, target):
        N, target_len = target.shape
        target_mask = torch.tril(torch.ones((target_len, target_len)).expand(N, 1, target_len,target_len))

        return target_mask.to(device)

    def forward(self, source, target):
        source_mask = self.make_source_mask(source)
        target_mask = self.make_target_mask(target)

        # encoder forward(self, x, mask)
        encoder_out = self.encoder(source, source_mask)

        # decoder forward(self, x, enc_out, source_mask, target_mask)
        out = self.decoder(target, encoder_out, source_mask, target_mask)

        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    seq_len = x.shape[1]

    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, seq_len, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)


# output : torch.Size([2, 7, 10])













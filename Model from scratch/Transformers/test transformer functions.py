import torch
import torch.nn as nn
from pos_encoding import PositionalEmbedding

a = torch.randn(3, 3)

# print(torch.tril(a))
#
# print(torch.tril(torch.ones((3, 3))).expand(3, 1, 3, 3))

x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]])
print(x[:, :-1].shape)

# class SelfAttention(nn.Module):
#     def __init__(self, embedding_size, num_heads):
#         """
#         Multi head attention
#         :param embedding_size: input embedding size
#         :param num_heads: number of split heads
#         """
#         super(SelfAttention, self).__init__()
#         self.embed_size = embedding_size
#         self.num_heads = num_heads
#         self.head_dims = self.embed_size // self.num_heads
#
#         assert (self.num_heads * self.head_dims == self.embed_size), "Embedding size should be divisible y number of heads"
#
#         # Linear layer for query
#         self.query_layer = nn.Linear(self.embed_size, self.embed_size, bias=False)  # If set to False, the layer will not learn an additive bias. Default: True
#         # Linear layer for Key
#         self.key_layer = nn.Linear(self.embed_size, self.embed_size, bias=False)
#         # Linear layer for Value
#         self.value_layer = nn.Linear(self.embed_size, self.embed_size, bias=False)
#
#         self.fc_out = nn.Linear(self.embed_size, self.embed_size)
#
#     def forward(self, queries, keys, values):
#         """
#         Suppose we have batch_size=32,sequence_length=10, embedding dimension=512. So after embedding and positional encoding our output will be of dimension 32x10x512.
#         We will resize it to 32x10x8x64.(About 8, it is the number of heads in multi head attention) and then bring it back to 32* 10 * 512
#         """
#         print(queries.shape)
#         N = queries.shape[0]
#         len_val, len_key, len_que = values.shape[1], keys.shape[1], queries.shape[1]  # seq_len for each matrix
#
#         queries = self.query_layer(queries)  # (N * len_que * embed_size)
#         values = self.value_layer(values)  # (N * len_val * embed_size
#         keys = self.key_layer(keys)  # (N * len_key * embed_size)
#
#         # adding explicit head dimensions : split last embed_size into num_heads * head_dims
#         values = values.reshape(N, len_val, self.num_heads, self.head_dims)  # 32 * 10 * 8 * 64
#         queries = queries.reshape(N, len_que, self.num_heads, self.head_dims)
#         keys = keys.reshape(N, len_key, self.num_heads, self.head_dims)
#
#         print(queries.shape)
#         print(keys.shape)
#
#
#         first_step = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])  # (N * num_heads, len_que, len_key)
#         print(first_step.shape)
#
#         #
#         # if mask is not None:
#         #     first_step = first_step.masked_fill(mask == 0, float("-1e20"))
#         #
#         # attention = torch.softmax(first_step / self.embed_size ** (1 / 2), dim=3)
#         # # Collapse the Head dimension by reshaping to (Batch, Sequence, Head * Query size). This effectively concatenates the Attention Score vectors for each head into a single merged Attention Score.
#         # out = torch.einsum([attention, values]).reshape(N, len_que, self.num_heads * self.head_dims)  # 32 * 10 * 512
#         #
#         # out = self.fc_out(out)  # (N * len_que, embed_size)
#         #
#         # return out
#
#
# embed_size = 512
# num_heads = 8
# word_embedding = nn.Embedding(10, embed_size)
# positional_embedding = PositionalEmbedding(seq_len=9, embed_size=embed_size)
# x = word_embedding(x)
#
# out = positional_embedding(x)
# # out= out.to(torch.float32)
# print()
# # print(x.shape)
# self_attention = SelfAttention(embed_size, num_heads)
# attention = self_attention(out, out, out)

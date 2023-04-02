# Positional Encoding in Transformer from scratch
# Steps involved

"""
L : input sequence length
k : position of an object in the input sequence, 0 <= k <= L/2
d : Dimension of output embedding space
P(k ,j) : position function for mapping a position k in the input sequence to index (k, j) of the positional matrix
n : user defined scaler, set to 10000 by the authors of attention
i : used for mapping to column indices 0 <= i <= d/2 or 0<= i <= d
"""

import numpy as np
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_size, n=10000):
        """

        :param seq_len: length of input sequence
        :param embed_size: embedding size
        """
        super(PositionalEmbedding, self).__init__()
        self.embed_size = embed_size
        self.max_len = seq_len
        pe = np.zeros((self.max_len, self.embed_size))
        for k in range(self.max_len):
            for i in np.arange(0, embed_size, 2):
                deno = np.power(n, 2 * i / embed_size)
                pe[k, i] = np.sin(k / deno)
                pe[k, i + 1] = np.cos(k / deno)
        pe = np.array(pe, dtype=np.float32)
        pe = torch.tensor(pe)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)  # If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.

    def forward(self, x):
        """

        :param x: Input embedding
        :return: Combine input with positional embedding and return
        """
        x = x + torch.autograd.Variable(self.pe[:, :self.max_len], requires_grad=False)
        return x


# def createPositionalEncoding(seq_len, d, n=10000):
#     pe = np.zeros((seq_len, d))
#     for p in range(seq_len):
#         for i in np.arange(int(d / 2)):
#             deno = np.power(n, 2 * i / d)
#             pe[k, 2 * i] = np.sin(p / deno)
#             pe[k, 2 * i + 1] = np.cos(p / deno)
#     return P
#
#
# p = createPositionalEncoding(4, 6)
# print('****** Position encoding with i : range(0,d/2) *******')
# print(p)
# print('****** Position encoding with i : range(0, d, 2) ******')
# p = createPositionalEncoding2(4, 6)
# print(p)

# ****** Position encoding with i : range(0,d/2) ******* d dimensional vector
# [[ 0.          1.          0.          1.          0.          1.        ]
#  [ 0.84147098  0.54030231  0.04639922  0.99892298  0.00215443  0.99999768]
#  [ 0.90929743 -0.41614684  0.0926985   0.99569422  0.00430886  0.99999072]
#  [ 0.14112001 -0.9899925   0.1387981   0.9903207   0.00646326  0.99997911]]
# ****** Position encoding with i : range(0, d, 2) ****** # d dimensional vector
# [[ 0.00000000e+00  1.00000000e+00  0.00000000e+00  1.00000000e+00 0.00000000e+00  1.00000000e+00]
#  [ 8.41470985e-01  5.40302306e-01  2.15443302e-03  9.99997679e-01 4.64158883e-06  1.00000000e+00]
#  [ 9.09297427e-01 -4.16146837e-01  4.30885605e-03  9.99990717e-01 9.28317767e-06  1.00000000e+00]
#  [ 1.41120008e-01 -9.89992497e-01  6.46325907e-03  9.99979113e-01 1.39247665e-05  1.00000000e+00]]

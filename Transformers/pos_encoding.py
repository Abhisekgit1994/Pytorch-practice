# Positional Encoding in Transformer from scratch
# Steps involved

"""
L : input sequence length
k : position of an object in the input sequence, 0 <= k <= L/2
d : Dimension of output embedding space
P(k ,j) : position function for mapping a position k in the input sequence to index (k, j) of the positional matrix
n : user defined scaler, set to 10000 by the authors of attention
i : used for mapping to column indices 0 <= i <= d/2
"""

import numpy as np


def createPositionalEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        print(k)
        for i in np.arange(int(d/2)):
            deno = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k/deno)
            P[k, 2*i + 1] = np.cos(k/deno)
    return P


p = createPositionalEncoding(4, 6)
print(p)

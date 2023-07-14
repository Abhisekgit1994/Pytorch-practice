import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import argmax, prepare_sequence, log_sum_exp


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embed_size, hidden_dim, num_layers):
        super(BiLSTM_CRF, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.start_tag = "<START>"
        self.end_tag = "<END>"
        self.tag_to_ix[self.start_tag] = len(self.tag_to_ix)
        self.tag_to_ix[self.end_tag] = len(self.tag_to_ix)
        self.target_size = len(self.tag_to_ix)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_dim//2, num_layers=self.num_layers, bidirectional=True)

        # map the output of the LSTM to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim, self.target_size)

        # matrix of transition parameters. Entry i,j is the score of transitioning to i from j
        self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size))
        # no transitions allowed to the beginning of the sentence
        self.transitions.data[self.tag_to_ix[self.start_tag, :]] = -10000
        # no transitions allowed from the end of the sentence
        self.transitions.data[:, self.tag_to_ix[self.end_tag]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # hidden state and cell state
        return torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2)

    def _forward_alg(self, features):

        # do the forward algorithm to compute the partition function
        init_alpha = torch.full((1, self.target_size), -10000.)

        # start tag has all the score
        init_alpha[0][self.tag_to_ix[self.start_tag]] = 0
        print(init_alpha)

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            for next_tag in range(self.target_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.target_size)








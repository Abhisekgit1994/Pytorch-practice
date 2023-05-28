import torch
from torch import nn


class CRF(nn.Module):
    def __init__(self, nb_labels, sos_tag, eos_tag, batch_first=True):
        super(CRF, self).__init__()
        self.nb_labels = nb_labels
        self.sos_tag = sos_tag
        self.eos_tag = eos_tag
        self.batch_first = batch_first

        """
        A kind of tensor that is to be considered a module parameter. tensor subclasses. When they are assigned as module attributes they are automatically added to the list
        of its parameters, and will appear in parameters() iterator. Assigning a tensor does not have such effect. This is because one might want to cache some temporary state,
        like last hidden state of RNN. 
        """
        self.transitions = nn.Parameter(torch.empty(self.nb_labels, self.nb_labels))

    def init_weights(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # no transitions allowed to the beginning of sentence
        self.transitions.data[:, self.sos_tag] = -10000.0
        # no transitions allowed from the end of sentence
        self.transitions.data[self.eos_tag,:] = 10000.0





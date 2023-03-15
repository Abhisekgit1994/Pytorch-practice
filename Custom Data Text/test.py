import torch
from torch.nn.utils.rnn import pad_sequence
x_seq = [torch.tensor([5, 18, 29]), torch.tensor([32, 100]), torch.tensor([699, 6, 9, 17])]
x_padded = pad_sequence(x_seq, batch_first=True, padding_value=0)
# x_padded = [[5, 18, 29, 0], [32, 100, 0, 0], [699, 6, 9, 17]]
print(x_padded)
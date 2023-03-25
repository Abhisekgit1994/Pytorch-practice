# data : json/csv/tsv files
# batching and padding : BucketIterator
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchtext
# from torchtext import BucketIterator
# import torchtext.legacy
# import torchtext.transforms as T
# from torchtext.datasets import AG_NEWS
# from torchtext.data.functional import sentencepiece_numericalizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

import spacy
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

SOURCE_LANG = 'de'
TARGET_LANG = 'en'

class Tokenize:
    def __init__(self):
        self.spacy_ger = spacy.load('de')
        self.spacy_en = spacy.load('en')


    def tokenize_german(self, text):
        return [tok.text for tok in self.spacy_ger.tokenizer(text)]

    def tokenize_english(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]


mask = torch.tril(torch.ones((5, 5)))==1
print(mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0)))
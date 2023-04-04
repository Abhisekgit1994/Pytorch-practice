import operator
import spacy
import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab, build_vocab_from_iterator

SOURCE_LANG = 'de'
TARGET_LANG = 'en'

# mask = torch.tril(torch.ones((5, 5)))==1
# print(mask.float().masked_fill(mask==0, float('-inf')).masked_fill(mask==1, float(0.0)))


data = [('Ein Mann mit geschminktem Gesicht und Hals sieht aus wie ein schwer blutender Verwundeter.', 'A male with makeup on his face and neck that looks like blood and wounds.'),
        ('Ein älterer Mann sieht einem jüngeren Mann zu, der verschiedene Speisen grillt.', 'An elderly man watches a younger man grill various foods.'),
        ('Ein Paar versucht, den Ort zu finden, zu dem es will.', 'A couple are trying to find the place they want to go.'),
        ('Ein Mann sitzt schreibend an einem Tisch, der auf dem Bürgersteig aufgestellt ist.', 'A man sits writing at a table set up on a sidewalk.')]

special_symbols = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']


def yield_tokens(data_iter):
    lang_idx = {SOURCE_LANG: 0, TARGET_LANG: 1}
    for data in data_iter:
        yield get_tokenizer('spacy', language='de_core_news_sm')(data[lang_idx['de']])


lde = [['Ein', 'Mann', 'mit', 'geschminktem', 'Gesicht', 'und', 'Hals', 'sieht', 'aus', 'wie', 'ein', 'schwer', 'blutender', 'Verwundeter', '.'],
       ['Ein', 'älterer', 'Mann', 'sieht', 'einem', 'jüngeren', 'Mann', 'zu', ',', 'der', 'verschiedene', 'Speisen', 'grillt', '.'],
       ['Ein', 'Paar', 'versucht', ',', 'den', 'Ort', 'zu', 'finden', ',', 'zu', 'dem', 'es', 'will', '.']]

# print(build_vocab_from_iterator(lde, min_freq=1, specials=special_symbols, special_first=True).get_stoi())
# print(build_vocab_from_iterator(yield_tokens(data), min_freq=1, specials=special_symbols, special_first=True).get_itos())


adds = operator.add
mul = operator.mul
div = operator.truediv


def sequential_ops(*operations):
    def func(x, y):
        for op in operations:
            x = op(x, y)
        return x

    return func


print(sequential_ops(adds, mul, div)(2, 3))

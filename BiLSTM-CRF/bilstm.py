import nltk
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
nltk.download('conll2002')

train = list(nltk.corpus.conll2002.iob_sents('ned.train'))
dev = list(nltk.corpus.conll2002.iob_sents('ned.testa'))
test = list(nltk.corpus.conll2002.iob_sents('ned.testb'))\

print(train[:3])


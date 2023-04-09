import pandas as pd
import pickle

with open('D:/Abhi/Hindi To English/train.pkl', 'rb') as file:
    train_data = pd.read_pickle(file)

with open('D:/Abhi/Hindi To English/dev.pkl', 'rb') as file:
    dev_data = pd.read_pickle(file)


def createDataIterator(dataframe, typ):
    train_iter = set()
    for idx, rows in dataframe.iterrows():
        tup = (rows.Hindi, rows.English)
        train_iter.add(tup)

    with open(typ + '_hindiToEnglish.pkl', 'wb') as file:
        pickle.dump(list(train_iter), file)


createDataIterator(train_data, typ='train')
createDataIterator(dev_data, typ='val')
del train_data
del dev_data

# Create vocab for first 150k rows of the data and save them in a pkl file so that we can directly use the vocab later without creating it.
"""
with open('train_hindiToEnglish.pkl', 'rb') as file:
    train_iter = pd.read_pickle(file)[:150000]
with open('val_hindiToEnglish.pkl', 'rb') as file:
    val_iter = pd.read_pickle(file)
for lang in [SOURCE_LANG, TARGET_LANG]:
    vocab_transform[lang] = build_vocab_from_iterator(yield_tokens(train_iter, lang), min_freq=3, specials=special_symbols, special_first=True, )
for lang in [SOURCE_LANG, TARGET_LANG]:
    vocab_transform[lang].set_default_index(UNK)
# saving the vocabs as pickle to save creation time
with open('vocab.pkl', 'wb') as file:
    pickle.dump(vocab_transform, file)

"""

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

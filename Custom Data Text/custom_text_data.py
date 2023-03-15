import os
import numpy as np
import pandas as pd
import spacy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm')


# convert text to numerical values
# vocabulary mapping to an index
# pytorch dataset to load
# setup padding for every batch (same sequence length)


class Vocabulary:
    def __init__(self, freq_threshold):
        self.idtostr = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.strtoid = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.idtostr)

    @staticmethod
    def tokenize_text(text):
        return [token.text.lower() for token in nlp.tokenizer(text)]

    def create_vocab(self, sentences):
        counter = {}
        idx = 4  # as we have used 0 to 3 for special tokens
        for sentence in sentences:
            tokens = self.tokenize_text(sentence)
            for token in tokens:
                if token not in counter:
                    counter[token] = 1
                else:
                    counter[token] += 1
                if counter[token] == self.freq_threshold:
                    self.strtoid[token] = idx
                    self.idtostr[idx] = token
                    idx = idx + 1

    def text_to_numeric(self, text):
        tokenized = self.tokenize_text(text)
        embedding = [self.strtoid[token] if token in self.strtoid else self.strtoid['<UNK>'] for token in tokenized]
        return embedding


class CustomFlickr(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=3):
        self.root_dir = root_dir
        self.df = pd.read_csv(caption_file)
        self.transform = transform

        # get image and caption columns
        self.images = self.df['image']
        self.captions = self.df['caption']

        # Initialize a vocabulary
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.create_vocab(self.captions.to_list())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        cap = self.captions[index]
        img_name = self.images[index]

        image = Image.open(os.path.join(self.root_dir, img_name)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption_embedding = [self.vocab.strtoid['<SOS>']]
        embed = self.vocab.text_to_numeric(cap)
        caption_embedding.extend(embed)
        caption_embedding.append(self.vocab.strtoid['<EOS>'])

        return image, torch.tensor(caption_embedding)


class Padding:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)
        return images, targets


def data_loader(root_dir, caption_file, transform, batch_size=32, num_workers=4, shuffle=True):
    dataset = CustomFlickr(root_dir, caption_file, transform=transform)
    pad_idx = dataset.vocab.strtoid['<PAD>']
    loader = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, collate_fn=Padding(pad_idx=pad_idx))
    return loader, dataset


def main():
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    loader, dataset = data_loader(root_dir='../datasets/caption data/Images/', caption_file='../datasets/caption data/captions.txt', transform=transform)

    for idx, (imgs, captions) in enumerate(loader):
        print(imgs.shape)
        print(captions.shape)


if __name__ == '__main__':
    main()

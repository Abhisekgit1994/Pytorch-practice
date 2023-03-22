import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import torch.nn.functional as fn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyper parameters
num_classes = 1000
in_channels = 3
lr = 0.001
batch = 8
epochs = 1


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        """

        :param embed_size: to make the final layer of pretrained model to match with embed size to concat with the 1st layer of lstm
        :param train_CNN: to fine tune freeze layers of pretrained model
        """
        super(EncoderCNN, self).__init__()
        self.embed_size = embed_size
        self.train_CNN = train_CNN
        self.inception = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception.fc = nn.Linear(self.inception.fc.in_features, self.embed_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        features = self.inception(images)
        for name, param in self.inception.named_parameters():
            if 'fc.weight' or 'fc.bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = self.train_CNN

        return self.dropout(self.relu(features))


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

        hiddens, _ = self.lstm(embeddings)
        output = self.linear(hiddens)

        return output


class EncoderToDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(EncoderToDecoder, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder= DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def pred_caption_image(self, image, vocabulary, max_length = 60):
        caption_predictions = []

        with torch.no_grad():
            x = self.encoder(image).unsqueeze(0)
            state = None # initial state
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, state)
                output = self.decoder.linear(hiddens.squeeze(0))  # taking the feature from encoder for the 1st layer of out decoder
                pred = output.argmax(1) # getting the 1st prediction from the decoder

                caption_predictions.append(pred.item())
                x = self.decoder.lstm(pred).unsqueeze(0)  # taking the previous output and feed as next hidden state

                if vocabulary.idtostr[pred.item()] == '<EOS>':
                    break

        return [vocabulary.idtostr[idx] for idx in caption_predictions]

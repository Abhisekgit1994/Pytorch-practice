import torch
import torch.nn as nn
import string
import random
import unidecode
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

all_chars = string.printable
# print(all_chars)
n_chars = len(all_chars)

file = unidecode.unidecode(open('names.txt').read())

# print(file)
names = [each.split(',')[0] for each in file.split('\n')]

names = '\n'.join(names)
print(all_chars.index('a'))


# print(names[5])


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embed = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        out = self.embed(x)  # make embedding of hidden size to feed into model next
        out, (hidden, cell) = self.lstm(out.unsqueeze(0), (hidden, cell))
        out = self.fc(out.reshape(out.shape[0], -1))
        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        return hidden, cell


class Generate(nn.Module):
    def __init__(self):
        super(Generate, self).__init__()
        self.lstm = None
        self.chunk_chars_len = 250  # how many characters can take at a time
        self.num_epochs = 100
        self.batch_size = 1
        self.print_interval = 10
        self.hidden_size = 256
        self.num_layers = 2
        self.lr = 0.003

    def char_tensor(self, string):
        """
        :param string: string of length 250 from the names file
        :return: a tensor after being replaced each character index taken from all_char
        """
        tensor = torch.zeros(len(string)).long()
        for c in range(len(string)):
            tensor[c] = all_chars.index(string[c])
        return tensor

    def get_random_batches(self):
        """
        Create random batch of 250 length from the names file to proceed with model training
        :return: text_input and target tensor of chunk_char_len length = 250
        """
        start_idx = random.randint(0, len(names) - self.chunk_chars_len)
        end_idx = start_idx + self.chunk_chars_len + 1  # length is 251 because 1st 250 for input and last 250 for target tensor
        text_str = names[start_idx:end_idx]
        text_input = torch.zeros(self.batch_size, self.chunk_chars_len)
        text_target = torch.zeros(self.batch_size, self.chunk_chars_len)

        for i in range(self.batch_size):
            text_input[i, :] = self.char_tensor(text_str[:-1])
            text_target[i, :] = self.char_tensor(text_str[1:])

        return text_input.long(), text_target.long()

    def generate(self, initial_str='Ab', predict_len=100, confidence=0.85):
        print("Generating New names")
        hidden, cell = self.lstm.init_hidden(batch_size=self.batch_size)
        initial_input = self.char_tensor(initial_str)
        predicted = initial_str

        # create hidden and cell for the length of initial given string so that we will use that to predict further
        for i in range(len(initial_str)-1):
            _, (hidden, cell) = self.lstm(initial_input[i].view(1).to(device), hidden, cell)

        # then take the last char as input to our model and generate new ones
        last_char = initial_input[-1]

        for i in range(predict_len):
            out, (hidden, cell) = self.lstm(last_char.view(1).to(device), hidden, cell)
            output_dist = out.data.view(-1).div(confidence).exp()
            top_char = torch.multinomial(output_dist, 1)[0]
            predict_char = all_chars[top_char]
            predicted += predict_char
            last_char = self.char_tensor(predict_char)

        return predicted

    def train(self):
        """
        # input_size : 1, hidden_size, num_layers, output_size
        :return: train the model
        """
        self.lstm = LSTM(n_chars, self.hidden_size, self.num_layers, n_chars).to(device)
        optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.lr)
        criteria = nn.CrossEntropyLoss()
        writer = SummaryWriter(f'runs/names0') # tensorboard

        print("***** Starting training *****")
        for e in range(1, self.num_epochs+1):
            inputs, targets = self.get_random_batches()  # length of 250 size

            # print(targets)
            hidden, cell = self.lstm.init_hidden(batch_size=self.batch_size)

            self.lstm.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            loss = 0
            # accumulate loss as the model running for each character prediction
            for i in range(self.chunk_chars_len):
                output, (hidden, cell) = self.lstm(inputs[:, i], hidden, cell)
                loss += criteria(output, targets[:, i])

            loss.backward()
            optimizer.step()
            loss = loss.item() / self.chunk_chars_len

            if e % self.print_interval == 0:
                print(f"loss : {loss}")
                # print(self.generate())

            writer.add_scalar('Training Loss', loss, global_step=e)


if __name__ == "__main__":
    model = Generate()
    model.train()
    print(model.generate())
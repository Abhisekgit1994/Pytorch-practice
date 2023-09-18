import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from utils import argmax, prepare_sequence, log_sum_exp


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embed_size, hidden_dim, num_layers=3):
        super(BiLSTM_CRF, self).__init__()
        self.embed_size = embed_size
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.start_tag = "<START>"
        self.end_tag = "<END>"
        self.tag_to_ix[self.start_tag] = len(self.tag_to_ix)
        self.tag_to_ix[self.end_tag] = len(self.tag_to_ix)
        self.target_size = len(self.tag_to_ix)

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_dim//2, num_layers=self.num_layers, bidirectional=True)

        # map the output of the LSTM to tag space
        self.hidden2tag = nn.Linear(self.hidden_dim, self.target_size)

        # matrix of transition parameters. Entry i,j is the score of transitioning from j to i
        self.transitions = nn.Parameter(torch.randn(self.target_size, self.target_size))
        # no transitions allowed to the beginning of the sentence
        self.transitions.data[self.tag_to_ix[self.start_tag], :] = -10000
        # no transitions allowed from the end of the sentence
        self.transitions.data[:, self.tag_to_ix[self.end_tag]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # hidden state and cell state
        return torch.randn(self.num_layers*2, 1, self.hidden_dim // 2), torch.randn(self.num_layers*2, 1, self.hidden_dim // 2)

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.embedding(sentence)
        embeds = self.embedding(sentence).unsqueeze(1)
        print(embeds.shape)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[self.start_tag]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
        score = score + self.transitions[self.tag_to_ix[self.end_tag], tags[-1]]
        return score

    def _forward_alg(self, features):

        # do the forward algorithm to compute the partition function
        init_alpha = torch.full((1, self.target_size), -10000.)
        # start tag has all the score
        init_alpha[0][self.tag_to_ix[self.start_tag]] = 0

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alpha
        # print(self.transitions)
        for feat in features:
            alphas_t = []
            for next_tag in range(self.target_size):
                # broadcast the emission score: it is the same regardless of the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.target_size)
                # print(next_tag)

                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                # print(next_tag_var.shape)
                log_sum = log_sum_exp(next_tag_var).view(1)

                alphas_t.append(log_sum)
            forward_var = torch.cat(alphas_t).view(1, -1)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.end_tag]]
        alpha = log_sum_exp(terminal_var)

        return alpha

    def _viterbi_decode(self, feats):
        backpointers = []

        # initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.target_size), -10000.)
        init_vvars[0][self.tag_to_ix[self.start_tag]] = 0
        print(init_vvars)

        # forward var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.target_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the previous step, plus the score of transitioning
                # from tag i next_tag
                # we don't include the emission scores here because the max does not depend on them
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[self.end_tag]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # follow the backpointer to decode the best path
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        # pop the start tag
        start = best_path.pop()
        assert start == self.tag_to_ix[self.start_tag]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


EMBEDDING_DIM = 10
HIDDEN_DIM = 4

# Make up some training data
training_data = [(
    "the wall street journal reported today that apple corporation made money".split(),
    "B I I I O O O B I O O".split()
), (
    "georgia tech is a university in georgia".split(),
    "B I O O O O B".split()
)]

word_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

tag_to_ix = {"B": 0, "I": 1, "O": 2}

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
load_model = False


def load_checkpoint(checkpoint):
    print("loading checkpoint......")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model.transitions)
    print(model(precheck_sent))


if load_model:
    load_checkpoint(torch.load('my_model.pth.tar'))
else:
    for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
        for sentence, tags in training_data:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Tensors of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            print(sentence_in.shape)
            targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
            print(targets)
            exit()
            # Step 3. Run our forward pass.
            loss = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # print(
            #     f"Epoch [{epoch}/{300}], "
            #     f"Loss: {loss.item():.4f}"
            # )
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(checkpoint, 'my_model.pth.tar')


# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model.transitions)
    print(model(precheck_sent))


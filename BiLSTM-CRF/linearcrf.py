import torch
import torch.nn as nn
from crf import CRF
import torch.optim as optim


class LinearCRF(nn.Module):
    def __init__(self, num_tags, sos_tag_id, eos_tag_id, batch_first=True):
        super(LinearCRF, self).__init__()
        self.num_tags = num_tags
        self.sos_tag = sos_tag_id
        self.eos_tag = eos_tag_id
        self.batch_first = batch_first
        self.transition = nn.Parameter(torch.empty(self.num_tags, self.num_tags))
        self.init_weights()

    def init_weights(self):

        nn.init.uniform_(self.transition, -0.1, 0.1)
        # enforce constraints (rows=from, columns=to) with a big negative number
        # so exp(-10000) will tend to zero

        self.transition.data[:, self.sos_tag] = -10000
        self.transition.data[self.eos_tag, :] = -10000

    def forward(self, emissions, tags, mask=None):
        nll = -self.log_likelihood(emissions, tags, mask=mask)
        return nll

    def log_likelihood(self, emissions, tags, mask=None):
        """

        :param emissions: sequence of emissions for each label, (batch, seq_len, num_tags)
        :param tags: sequence of labels, (batch, seq_len)
        :param mask: tensor representing valid positions, (batch, seq_len)
        :return: torch.Tensor: the summed log-likelihood of each sequence in the batch. shape of (1,)
        """

        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask=mask)
        partition = self._compute_log_partition(emissions, mask=mask)

        return torch.sum(scores - partition)

    def _compute_scores(self, emissions, tags, mask):
        """

        :param emissions: (batch, seq_len, num_tags)
        :param tags: (batch, seq_len)
        :param mask: (batch, seq_len)
        :return: scores of each batch. (batch,)
        """

        batch_size, seq_len = tags.shape
        scores = torch.zeros(batch_size)

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # add the transition from sos to the first tags for each batch
        t_scores = self.transition[self.sos_tag, first_tags]

        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_nlabels]]

        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()

        scores += e_scores + t_scores

        for i in range(1, seq_len):
            # we could: iterate over batches, check if we reached a mask symbol
            # and stop the iteration, but vectorizing is faster due to gpu,
            # so instead we perform an element wise multiplication

            is_valid = mask[:, i]

            previous_tags = tags[:, i-1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self.transition[previous_tags, current_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        scores += self.transition[last_tags, self.eos_tag]

        return scores

    def _compute_log_partition(self, emissions, mask):

        batch, seq_len, num_tags = emissions.shape

        alphas = self.transition[self.sos_tag, :].unsqueeze(0) + emissions[:, 0]

        for i in range(1, seq_len):
            alpha_t = []

            for tag in range(num_tags):

                # get the emission for the current tag
                e_scores = emissions[:, i, tag]

                # broadcast emission to all labels since it will be the same for all previous tags
                e_scores = e_scores.unsqueeze(1)

                # transitions from something to our tag
                t_scores = self.transition[:, tag]

                # broadcast the transition score to all batches
                t_scores = t_scores.unsqueeze(0)

                scores = e_scores + t_scores + alphas

















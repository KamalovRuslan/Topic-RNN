import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from topic_rnn_optimizer import TRnnOpt
import storage


use_cuda = torch.cuda.is_available()


class TopicRNN(nn.Module):

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, ntopics, dropout=0.5):
        super.(TopicRNN, self).__init__()

        self.drop = nn.Dropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            raise ValueError("rnn_type should be GRU or LSTM!")

        self.vocab_dot_topics = nn.Linear(ntoken, ntopics)
        self.decoder = nn.Linear(nhid, ntopics)

        try:
            self.init_weights()
        except AttributeError:
            pass

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.ntopics = ntopics

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.vocab_dot_topics.weight.data.uniform_(0, 1)
        self.vocab_dot_topics = F.normalize(self.wt, p=1, dim=1)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        idx, input = input
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        w_dist = torch.dot(self.vocab_dot_topics[idx, :], decoded)
        return w_dist, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())

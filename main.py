from TopicRNN import TopicRNN
from storage import Dictionary, Corpus
import torch
from torch.autograd import Variable
from TopicOptimizer import TRnnOpt

PATH = './data/train.txt'

corpus = Corpus(PATH)
RNNTYPE = 'GRU'
epochs = 500


def batchify(corpus):
    # TODO
    yield batch


def loss_function(model, idx, out):
    loss = model.TokTop[idx](out)
    # TODO
    return loss


def test(epoch):
    # TODO
    return loss


def train(epoch, optimizer):
    for batch in butchify(corpus):
        batch = Variable(batch)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        out = model(data)
        loss = loss_function(out)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        model.normalize_phi()

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(corpus)))


for epoch in range(1, epochs):
    model = TopicRNN(RNNTYPE, corpus.shape[0])
    model.train()
    exp_group = {'exp_group': [layer.parameters() for layer in model.TokTop.values()]}
    com_group = {'com_group': [model.encoder.parameters(), model.decoder.parameters()]}
    optimizer = TRnnOpt([exp_group, com_group])
    train(epoch)
    # TODO

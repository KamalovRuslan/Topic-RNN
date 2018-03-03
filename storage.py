import torch
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer


class Dictionary(object):

    def __init__(self):
        word2idx = dict()
        idx2word = []
        stemmer = SnowballStemmer('english')

    def _add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(stemmer(word))
            self.word2idx[word] = len(self.idx2word) - 1
        return

    def add_sentence(self, sentence):
        tokens = 0
        for word in sentence.strip().split(' '):
            self._add_word(word)
            tokens += 1
        return tokens

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):

    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self._tokenize(os.path.join(path, 'train.txt'))
        self.valid = self._tokenize(os.path.join(path, 'valid.txt'))
        self.test = self._tokenize(os.path.join(path, 'test.txt'))

    def _tokenize(self, path):
        assert os.path.exists(path)

        with open(path, 'r') as f:
            text = f.read()
        sentences = tokenize.sent_tokenize(text)
        tokens = 0
        for sentence in sentences:
            tokens += dictionary.add_sentence(sentence)

        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for sentence in sentences:
                words = line.strip().split()
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        return ids

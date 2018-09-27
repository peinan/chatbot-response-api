import numpy as np

from vectorizers.vectorizerbase import VectorizerBase
from utils import util as u


class VectorizerW2V(VectorizerBase):
    def __init__(self):
        super(VectorizerW2V, self).__init__()
        self.model = u.Util.load_model('word2vec')

    def lookup(self, word: str) -> np.array:
        try:
            word_vec = self.model.wv[word]
        except KeyError:
            word_vec = np.array([])

        return word_vec

    def sent2vec(self, sentence_or_words, mode='average') -> np.array:
        if type(sentence_or_words) == str:
            words = self.mparser.wakati(sentence_or_words)
        elif type(sentence_or_words) == list:
            words = sentence_or_words
        else:
            raise ValueError(f'"{sentence_or_words}" is neither a list nor a str.')

        word_vecs = [ self.lookup(word) for word in words if len(self.lookup(word)) != 0 ]
        sentvec = np.nanmean(np.array(word_vecs), axis=0)

        if np.all(np.isnan(sentvec)):
            sentvec = np.zeros(self.model.wv.vector_size)
            sentvec[:] = np.nan

        return sentvec

    def make_sentmtx(self, sentences: list) -> np.array:
        sentmtx = np.array([ self.sent2vec(s) for s in sentences])

        return sentmtx


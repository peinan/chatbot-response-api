from vectorizers.vectorizerbase import VectorizerBase
import numpy as np

from utils import util as u


class VectorizerCharNgram(VectorizerBase):
    def __init__(self):
        super(VectorizerCharNgram, self).__init__()

    @staticmethod
    def ngram_split(sentence, n=2) -> list:
        ngrams = []
        sent_len = len(sentence)

        assert n > 0, f'n={n} must larger than 0'
        if n > sent_len:
            return []

        for i in range(sent_len - n + 1):
            ngram = ''
            for j in range(n):
                char = sentence[i + j]
                ngram += char

                ngrams.append(ngram)

        return u.Util.add_BOS_EOS(ngrams)

    def sent2vec(self, sentence_or_words, n=2) -> np.array:
        if type(sentence_or_words) == str:
            sentence = sentence_or_words
        elif type(sentence_or_words) == list:
            sentence = ''.join(sentence_or_words)
        else:
            raise ValueError(f'"{sentence_or_words}" is neither a list nor a str.')

        ngrams = self.ngram_split(sentence, n=n)
        sentence_minhash = u.Util.minhash(ngrams)

        return sentence_minhash

    @staticmethod
    def make_sentmtx(sentences: list, n=2) -> np.array:
        """
        A new fast method using Feature Hashing
        :param sentences:
        :param n:
        :return:
        """

        return np.array([ u.Util.minhash(VectorizerCharNgram.ngram_split(sentence, n=n))
                          for sentence in sentences ])

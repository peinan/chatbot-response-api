from vectorizers.vectorizerbase import VectorizerBase
import numpy as np
import pickle as pkl

from utils import util as u


class VectorizerCharNgram(VectorizerBase):
    def __init__(self):
        super(VectorizerCharNgram, self).__init__()
        self.charngram2id = u.Util.load_data('charngram2id')

    def lookup(self, ngram: str) -> int:
        try:
            ngram_id = self.charngram2id[ngram]
        except KeyError:
            ngram_id = np.nan

        return ngram_id

    def sent2vec(self, sentence_or_words, n=2, flatten=True):
        """
        A method that make a sentence vector.

        :param sentence_or_words:
        :param n:
        :param flatten: make a 1-D vector
        :return: a list of ngram_ids or a numpy.array if the flatten option is True
        """

        if type(sentence_or_words) == str:
            sentence = sentence_or_words
        elif type(sentence_or_words) == list:
            sentence = ''.join(sentence_or_words)
        else:
            raise ValueError(f'"{sentence_or_words}" is neither a list nor a str.')

        ngrams = self.ngram_split(sentence, n=n)
        ngram_ids = [ self.lookup(ngram) for ngram in ngrams ]

        if flatten:
            flattened = np.zeros(len(self.charngram2id), dtype=np.int8)
            for ngram_id in ngram_ids:
                if not np.isnan(ngram_id):
                    flattened[ngram_id] += 1

            return flattened

        return ngram_ids

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

    @staticmethod
    def make_sentmtx(sentences: list, n=2) -> (np.array, dict, dict):
        """
        A preprocess of making sentence matrix with char-ngram.
        :param sentences: a list of sentences with string inside
        :param n: default is 2
        :return: sentmtx, ngram2id, id2ngram
        """
        ngramid_sentences = []
        ngram2id = {}
        id2ngram = {}
        for sentence in sentences:
            ngrams = VectorizerCharNgram.ngram_split(sentence, n=n)
            ngram_ids = []
            for ngram in ngrams:
                if ngram in ngram2id:
                    ngram_ids.append(ngram2id[ngram])
                else:
                    ngram_id = len(ngram2id)
                    ngram2id[ngram] = ngram_id
                    id2ngram[ngram_id] = ngram
                    ngram_ids.append(ngram_id)
            ngramid_sentences.append(ngram_ids)

        n_row = len(sentences)
        n_col = len(ngram2id)
        print(f'Char-{n}Gram model size will be ({n_row}, {n_col})')
        sentmtx = np.zeros((n_row, n_col), dtype=np.int8)
        for i, ngram_ids in enumerate(ngramid_sentences):
            for ngram_id in ngram_ids:
                sentmtx[i, ngram_id] += 1

        return sentmtx, ngram2id, id2ngram

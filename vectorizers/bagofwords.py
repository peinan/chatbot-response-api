from vectorizers.vectorizerbase import VectorizerBase
import numpy as np
import pickle as pkl


class VectorizerBagOfWords(VectorizerBase):
    def __init__(self):
        super(VectorizerBagOfWords, self).__init__()
        self.word2id = pkl.load(open(self.word2id_path, 'rb'))

    def lookup(self, word: str) -> int:
        try:
            word_id = self.word2id[word]
        except KeyError:
            word_id = -1

        return word_id

    def sent2vec(self, sentence_or_words, n=2, mode='id', flatten=True):
        """
        A method that make a sentence vector.

        :param sentence_or_words:
        :param n:
        :param mode: "id" or "word"
        :param flatten: make a 1-D vector
        :return: a list of ngrams or a numpy.array if the flatten option is True
        """

        if type(sentence_or_words) == str:
            words = self.mparser.wakati(sentence_or_words)
        elif type(sentence_or_words) == list:
            words = sentence_or_words
        else:
            raise ValueError(f'"{sentence_or_words}" is neither a list nor a str.')

        ngrams = []
        n_words = len(words)

        assert n > 0, f'n={n} must larger than 0'
        assert n <= n_words, f'n={n} must smaller than sentence length'

        for i in range(n_words - n + 1):
            ngram = []
            for j in range(n):
                word = n_words[i + j]
                if mode == 'id':
                    word_id = self.lookup(word)
                    if word_id != -1:
                        ngram.append(word_id)

            ngrams.append(ngram)

        if mode == 'id' and flatten:
            flattened = np.zeros(len(self.word2id), dtype=np.int8)
            for ngram in ngrams:
                for word_id in ngram:
                    flattened[word_id] += 1

            return flattened

        return ngrams

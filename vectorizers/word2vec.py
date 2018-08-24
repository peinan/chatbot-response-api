from vectorizers.vectorizerbase import VectorizerBase
from gensim.models import word2vec
import numpy as np
from typing import Optional


class VectorizerW2V(VectorizerBase):
    def __init__(self):
        super(VectorizerW2V, self).__init__()
        self.model = word2vec.Word2Vec.load(self.w2v_model_path)

    def lookup(self, word: str) -> np.array:
        try:
            word_vec = self.model.wv[word]
        except KeyError:
            word_vec = np.array([])

        return word_vec

    def sent2vec(self, sentence_or_words, mode='average') -> Optional[np.array]:
        if type(sentence_or_words) == str:
            words = self.mparser.wakati(sentence_or_words)
        elif type(sentence_or_words) == list:
            words = sentence_or_words
        else:
            raise ValueError(f'"{sentence_or_words}" is neither a list nor a str.')

        if mode == 'average':
            word_vecs = [ self.lookup(word) for word in words if len(self.lookup(word)) ]
            sent_vec = np.array(word_vecs).mean(axis=0)
            if not len(sent_vec.shape):
                sent_vec = np.zeros(self.model.wv.vector_size)
        else:
            sent_vec = np.zeros(self.model.wv.vector_size)

        return sent_vec

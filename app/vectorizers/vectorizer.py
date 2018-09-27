from vectorizers.vectorizerbase import VectorizerBase
from vectorizers.word2vec import VectorizerW2V
from vectorizers.charngram import VectorizerCharNgram


class Vectorizer(VectorizerBase):
    def __init__(self):
        super(Vectorizer, self).__init__()
        self.load_all_vectorizers()

    def load_all_vectorizers(self):
        self.vectorizer_w2v = VectorizerW2V()
        self.vectorizer_charngram = VectorizerCharNgram()

    def vectorize(self, utterance: str, mode='all') -> dict:
        words = self.mparser.wakati(utterance)

        if mode == 'word2vec':
            vecs = {'word2vec': self.vectorizer_w2v.sent2vec(words)}
        elif mode == 'charngram':
            vecs = {'charngram': self.vectorizer_charngram.sent2vec(words)}
        else:
            vecs = {
                'word2vec': self.vectorizer_w2v.sent2vec(words),
                'charngram': self.vectorizer_charngram.sent2vec(words)
            }

        return vecs

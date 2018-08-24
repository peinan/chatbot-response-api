import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from utils.util import Util


class Evaluator:
    def __init__(self):
        self.qas = Util.load_qas()
        self.sentmtx = Util.load_sentmtx()

    def _calc_similarities(self, tgt_vec: np.array) -> np.array:
        assert tgt_vec.shape == (self.sentmtx.shape[1],)

        similarities = cosine_similarity(self.sentmtx, tgt_vec.reshape(1, -1))

        return similarities

    def get_topn_replies(self, uttr_vec: np.array, topn: int) -> list:
        similarities = self._calc_similarities(uttr_vec)
        sorted_topn_indices = similarities.argsort(axis=0)[::-1][:topn, 0]
        replies = []
        for idx in sorted_topn_indices:
            q, a = self.qas[idx]
            replies.append(a)

        return replies

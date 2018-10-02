import numpy as np
import time

from utils import util as u
from logging import getLogger


class Evaluator:
    def __init__(self):
        self.qas = u.Util.load_data('qas')
        self.w2v_sentmtx = u.Util.load_data('word2vec')
        self.charngram_sentmtx = u.Util.load_data('charngram')
        self.logger = getLogger('doppel')

    def _calc_w2v_sim(self, tgt_vec: np.array) -> np.array:
        assert tgt_vec.shape == (self.w2v_sentmtx.shape[1],)

        sim = u.Util.cosine_similarity(self.w2v_sentmtx, tgt_vec.reshape(1, -1))

        return sim

    def _calc_charngram_sim(self, tgt_vec: np.array) -> np.array:
        assert tgt_vec.shape == (self.charngram_sentmtx.shape[1],)

        sim = u.Util.jaccard_similarity_minhash(self.charngram_sentmtx, tgt_vec.reshape(1, -1))

        return sim

    def _calc_sim(self, *sims) -> np.array:
        sim_ary = np.array(sims)
        assert sim_ary.shape == (len(sims), sims[0].shape[0])

        return np.nanmean(sim_ary, axis=0)

    def get_topn_replies(self, uttr_vecs: dict, topn: int) -> list:

        cp1 = time.time()

        w2v_sim = self._calc_w2v_sim(uttr_vecs['word2vec'])

        cp2 = time.time()

        charngram_sim = self._calc_charngram_sim(uttr_vecs['charngram'])

        cp3 = time.time()

        sims = self._calc_sim(w2v_sim, charngram_sim)

        cp4 = time.time()

        sorted_topn_indices = sims.argsort(axis=0)[(sims.shape[0] - topn):][::-1]

        cp5 = time.time()

        replies = []
        for idx in sorted_topn_indices:
            q, a = self.qas[idx]
            replies.append({'text': a,
                            'nearest_q': q,
                            'similarity': sims[idx],
                            'word2vec_sim': w2v_sim[idx],
                            'charngram_sim': charngram_sim[idx]})

        cp6 = time.time()

        self.logger.info(f'Result: {replies}')
        self.logger.info(f'Time: [Total] {cp6-cp1:.3f}s, '
                         f'[CP1-2] {cp2-cp1:.3f}s, '
                         f'[CP2-3] {cp3-cp2:.3f}s, '
                         f'[CP3-4] {cp4-cp3:.3f}s, '
                         f'[CP4-5] {cp5-cp4:.3f}s, '
                         f'[CP5-6] {cp6-cp5:.3f}s')

        return replies

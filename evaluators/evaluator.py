import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

from utils import util as u

import time


class Evaluator:
    def __init__(self):
        self.qas = u.Util.load_data('qas')
        self.w2v_sentmtx = u.Util.load_data('word2vec')
        self.charngram_sentmtx = u.Util.load_data('charngram')

    def _calc_w2v_sim(self, tgt_vec: np.array) -> np.array:
        assert tgt_vec.shape == (self.w2v_sentmtx.shape[1],)

        sim = u.Util.cosine_similarity(self.w2v_sentmtx, tgt_vec.reshape(1, -1))

        return sim

    def _calc_charngram_sim(self, tgt_vec: np.array) -> np.array:
        assert tgt_vec.shape == (self.charngram_sentmtx.shape[1],)

        sim = u.Util.jaccard_similarity(self.charngram_sentmtx, tgt_vec.reshape(1, -1))

        return sim

    def _calc_sim(self, *sims) -> np.array:
        sim_ary = np.array(sims)
        assert sim_ary.shape == (len(sims), sims[0].shape[0])

        return np.nanmean(sim_ary, axis=0)

    def get_topn_replies(self, uttr_vecs: dict, topn: int) -> list:

        cp1 = time.time()

        w2v_sim = self._calc_w2v_sim(uttr_vecs['word2vec'])

        cp2 = time.time()
        print(f'Lap CP2: {cp2-cp1:.4f}s')

        charngram_sim = self._calc_charngram_sim(uttr_vecs['charngram'])

        cp3 = time.time()
        print(f'Lap CP3: {cp3-cp2:.4f}s')

        sims = self._calc_sim(w2v_sim, charngram_sim)

        cp4 = time.time()
        print(f'Lap CP4: {cp4-cp3:.4f}s')

        sorted_topn_indices = sims.argsort(axis=0)[(sims.shape[0] - topn):]

        cp5 = time.time()
        print(f'Lap CP5: {cp5-cp4:.4f}s')

        replies = []
        for idx in sorted_topn_indices:
            q, a = self.qas[idx]
            replies.append((a,
                            {'word2vec_sim': w2v_sim[idx],
                             'charngram_sim': charngram_sim[idx]}))

        cp6 = time.time()
        print(f'Lap CP6: {cp6-cp5:.4f}s')

        print(f'Total: {cp6-cp1:.4f}s')

        return replies

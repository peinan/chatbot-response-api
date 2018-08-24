import numpy as np
import pickle as pkl
from os import path
import sys

from db.manager import DBCrawledManager
from utils.morphparser import MorphParser
from vectorizers.word2vec import VectorizerW2V


def load_qas(n_max=10):
    crawled_db = DBCrawledManager()
    qas = crawled_db.fetch_exist_qas(n_max)

    return qas


def load_vectorizer():
    vectorizer = VectorizerW2V()

    return vectorizer


def load_morph_parser():
    mparser = MorphParser()

    return mparser


def make_sentence_matrix(sentences, mparser, vectorizer) -> (np.array, list):
    sent_matrix = []
    sent_idxs = []
    for i, sentence in enumerate(sentences):
        sent_vec = vectorizer.sent2vec(mparser.wakati(sentence))
        if sent_vec.any():
            sent_matrix.append(sent_vec)
            sent_idxs.append(i)

    return np.array(sent_matrix), sent_idxs


def main(n_max=10):
    qas = load_qas(n_max)

    questions = [ qa[0] for qa in qas ]

    vectorizer = load_vectorizer()
    mparser = load_morph_parser()

    sent_matrix, sent_idxs = make_sentence_matrix(questions, mparser, vectorizer)
    avaliable_qas = [ qas[idx] for idx in sent_idxs ]

    datadir = path.dirname(path.abspath(__file__)) + '/../data'
    sent_matrix_filepath = datadir + '/sentence_matrix.pkl'
    qas_filepath = datadir + '/qas.pkl'
    pkl.dump(sent_matrix, open(sent_matrix_filepath, 'wb'))
    pkl.dump(avaliable_qas, open(qas_filepath, 'wb'))


if __name__ == '__main__':
    try:
        n_max = sys.argv[1]
        main(int(n_max))
    except:
        main()

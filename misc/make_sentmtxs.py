from pathlib import Path
from vectorizers.charngram import VectorizerCharNgram
from vectorizers.word2vec import VectorizerW2V
from utils import util as u


def make_sentmtx_charngram(qs, datadir):
    """
    A new fast method using Feature Hasing
    :param qs:
    :param datadir:
    :return:
    """

    sentmtx_c2g = VectorizerCharNgram.make_sentmtx(qs)

    u.Util.pickle_dump(sentmtx_c2g, f"{datadir / 'sentmtx.charngram.pkl'}")


def make_sentmtx_word2vec(qs, datadir):
    # make word2vec sentmtx
    vectorizer_w2v = VectorizerW2V()
    sentmtx_w2v = vectorizer_w2v.make_sentmtx(qs)

    # dump
    u.Util.pickle_dump(sentmtx_w2v, f"{datadir / 'sentmtx.word2vec.pkl'}")


def main():
    datadir = Path(__file__).absolute().parent / '../data'
    qas_filepath = datadir / 'raw_qas.pkl'
    qas = u.Util.pickle_load(qas_filepath)
    qs = [ qa[0] for qa in qas ]

    make_sentmtx_charngram(qs, datadir)
    make_sentmtx_word2vec(qs, datadir)


if __name__ == '__main__':
    main()

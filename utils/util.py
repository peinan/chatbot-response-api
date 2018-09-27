import configparser
from pathlib import Path
import pickle as pkl
import numpy as np
from hashlib import sha1
import time, struct

from db.manager import DBCrawledManager


class MacOSFile(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


class Util:

    @staticmethod
    def pickle_dump(obj, file_path):
        with open(file_path, "wb") as f:
            return pkl.dump(obj, MacOSFile(f), protocol=pkl.HIGHEST_PROTOCOL)

    @staticmethod
    def pickle_load(file_path):
        with open(file_path, "rb") as f:
            return pkl.load(MacOSFile(f))

    @staticmethod
    def load_config(config_filepath=None):
        if not config_filepath:
            config_filename = 'config.ini'
            config_filepath = Path(__file__).absolute().parent / f'../etc/{config_filename}'
        print(f'Loading {config_filepath}')
        config = configparser.ConfigParser()
        config.read(config_filepath)

        return config

    @staticmethod
    def load_data(dataname):
        if dataname == 'word2vec':
            filename = 'sentmtx.word2vec.pkl'
        elif dataname == 'charngram':
            filename = 'sentmtx.charngram.pkl'
        elif dataname == 'charngram2id':
            filename = 'charngram2id.pkl'
        elif dataname == 'qas':
            filename = 'raw_qas.pkl'
        else:
            raise ValueError(f'Type {dataname} not found.')

        filepath = Path(__file__).absolute().parent / f'../data/{filename}'

        print(f'Loading {dataname}: {filepath}')

        data = Util.pickle_load(filepath)

        return data

    @staticmethod
    def cosine_similarity(x1: np.array, x2: np.array) -> np.array:
        assert len(x1.shape) == len(x2.shape) == 2
        assert x2.shape[0] == 1
        assert x1.shape[1] == x2.shape[1], 'it must be same dimension 2-D matrice.'

        num = x1 @ x2.reshape(x2.shape[1])
        den = np.linalg.norm(x1, axis=1) * np.linalg.norm(x2, axis=1)

        return num / den

    @staticmethod
    def normalize(x: np.array, axis=None) -> np.array:
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x - min) / (max - min)

        return result

    @staticmethod
    def jaccard_similarity(x1: np.array, x2: np.array) -> np.array:
        assert len(x1.shape) == len(x2.shape) == 2
        assert x1.shape[1] == x2.shape[1], 'it must be same dimension 2-D matrice.'

        cp1 = time.time()

        # 集合なので 2 以上の数を 1 にする
        x1[x1 >= 1] = 1
        x2[x2 >= 1] = 1

        cp2 = time.time()
        print(f'JS Lap CP2: {cp2-cp1:.4f}')

        x1_and_x2 = (x1 * x2).sum(axis=1)

        cp3 = time.time()
        print(f'JS Lap CP3: {cp3-cp2:.4f}')

        x_sum = x1 + x2

        cp4 = time.time()
        print(f'JS Lap CP4: {cp4-cp3:.4f}')

        x_sum[x_sum >= 1] = 1

        cp5 = time.time()
        print(f'JS Lap CP5: {cp5-cp4:.4f}')

        x1_or_x2 = x_sum.sum(axis=1)

        cp6 = time.time()
        print(f'JS Lap CP6: {cp6-cp5:.4f}')

        return x1_and_x2 / x1_or_x2

    @staticmethod
    def jaccard_similarity_minhash(x1: np.array, x2: np.array) -> np.array:
        assert len(x1.shape) == len(x2.shape) == 2
        assert x1.shape[1] == x2.shape[1]
        assert x2.shape[0] == 1

        return np.count_nonzero(x1==x2, axis=1).astype(np.float) / np.float(x1.shape[1])

    @staticmethod
    def minhash(ngrams: list) -> np.array:
        # init hashvalues
        _max_hash = (1 << 32) - 1
        _num_perm = 128
        _mersenne_prime = (1 << 61) - 1
        hashvalues = np.ones(_num_perm, dtype=np.uint64) * _max_hash

        # create generator
        seed = 1
        generator = np.random.RandomState(seed=seed)
        permutations = np.array([ (generator.randint(1, _mersenne_prime, dtype=np.uint64),
                                   generator.randint(0, _mersenne_prime, dtype=np.uint64))
                                  for _ in range(_num_perm)], dtype=np.uint64).T

        # update hashvalues
        hashobj = sha1
        for ngram in ngrams:
            b = ngram.encode('utf-8')
            hv = struct.unpack('<I', hashobj(b).digest()[:4])[0]
            a, b = permutations
            phv = np.bitwise_and((a * hv + b) % _mersenne_prime, np.uint64(_max_hash))
            hashvalues = np.minimum(phv, hashvalues)

        return hashvalues

    @staticmethod
    def add_BOS_EOS(items: list, BOS='<s>', EOS='</s>'):
        result_s = [BOS] + items + [EOS]

        return result_s

    @staticmethod
    def dump_raw_qas(dump_filepath, n_max=10):
        db = DBCrawledManager()
        raw_qas = db.fetch_exist_qas(n_max)

        pkl.dump(raw_qas, open(dump_filepath, 'wb'))

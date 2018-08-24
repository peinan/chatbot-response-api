import configparser
import os
import pickle as pkl


class Util:

    @staticmethod
    def load_config(config_filepath=None):
        if not config_filepath:
            config_filename = 'config.ini'
            config_filepath = os.path.dirname(os.path.abspath(__file__))\
                              + f'/../etc/{config_filename}'
        print(f'Loading {config_filepath}')
        config = configparser.ConfigParser()
        config.read(config_filepath)

        return config

    @staticmethod
    def load_qas(qas_filepath=None):
        if not qas_filepath:
            qas_filename = 'qas.pkl'
            qas_filepath = os.path.dirname(os.path.abspath(__file__))\
                              + f'/../data/{qas_filename}'
        print(f'Loading {qas_filepath}')

        qas = pkl.load(open(qas_filepath, 'rb'))

        return qas

    @staticmethod
    def load_sentmtx(sentmtx_filepath=None):
        if not sentmtx_filepath:
            sentmtx_filename = 'sentmtx.pkl'
            sentmtx_filepath = os.path.dirname(os.path.abspath(__file__)) \
                           + f'/../data/{sentmtx_filename}'
        print(f'Loading {sentmtx_filepath}')

        sentmtx = pkl.load(open(sentmtx_filepath, 'rb'))

        return sentmtx

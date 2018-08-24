import MeCab

from utils.util import Util


class MorphParser:
    def __init__(self, dict_filepath=None):
        self.config = Util.load_config()

        self.dict_filepath = dict_filepath
        if not dict_filepath:
            self.dict_filepath = self.config['mecab']['dict_filepath']

        self.tagger = MeCab.Tagger(f'-d {self.dict_filepath}')

    def parse(self, text):
        parsed_str = self.tagger.parse(text)
        lines = parsed_str.split('\n')
        result = []
        for line in lines:
            if line == 'EOS':
                break
            surf, feats = line.split('\t')
            result.append({
                'surface': surf,
                'features': tuple(feats.split(','))
            })

        return result

    def wakati(self, text):
        return [ res['surface'] for res in self.parse(text) ]

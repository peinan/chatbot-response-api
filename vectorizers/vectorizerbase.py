from utils.util import Util
from utils.morphparser import MorphParser


class VectorizerBase:
    def __init__(self):
        self.config = Util.load_config()
        self.w2v_model_path = self.config['models']['word2vec']
        self.mparser = MorphParser()

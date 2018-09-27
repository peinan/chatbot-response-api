from utils.util import Util
from utils.morphparser import MorphParser


class VectorizerBase:
    def __init__(self):
        self.config = Util.load_config()
        self.mparser = MorphParser()

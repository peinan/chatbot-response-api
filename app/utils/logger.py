import logging
import logging.handlers
from pathlib import Path


class Logger:

    def __init__(self, **kwargs):

        self.dirpath = Path(__file__).resolve().parent / '../logs'
        self.filename = 'application.log'
        self.level = logging.INFO
        self.name = None

        self.fmt = '[%(asctime)s][%(levelname)s][%(funcName)s][%(filename)s:%(lineno)s] %(message)s'

        if 'dirpath' in kwargs:
            self.dirpath= Path(kwargs['dirpath'])
        if 'filename' in kwargs:
            self.filename = kwargs['filename']
        if 'level' in kwargs:
            self.filename = kwargs['level']
        if 'name' in kwargs:
            self.name = kwargs['name']

        # make dir and files for logging
        self.dirpath.mkdir(parents=True, exist_ok=True)

        self.logfilepath = self.dirpath / self.filename
        self.__set_logger()

    def __set_logger(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.level)
        formatter = logging.Formatter(self.fmt)

        # stream handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(self.level)
        self.logger.addHandler(stream_handler)

        # file handler (w/ log rotate)
        file_handler = logging.handlers.RotatingFileHandler(self.logfilepath,
                                                            maxBytes=1048576,
                                                            backupCount=3)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self.level)
        self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger

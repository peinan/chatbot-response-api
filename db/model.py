import mysql.connector
from utils.util import Util


class DBCrawled:
    def __init__(self):
        self.config = Util.load_config()
        self.db_config = self.config['db_crawled']

    def make_connection(self):
        connection = mysql.connector.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            database=self.db_config['database']
        )

        return connection

from db.model import DBCrawled


class DBCrawledManager:
    def __init__(self):
        self.connection = DBCrawled().make_connection()
        self.cursor = self.connection.cursor()

    def fetch_exist_qas(self, n_max=10):
        stmt = (
            'SELECT peingQuestion, peingAnswer FROM peing '
            'WHERE peingQuestion <> "" AND peingAnswer <> ""'
        )
        if n_max:
            stmt += f' LIMIT {n_max}'

        self.cursor.execute(stmt)

        return self.cursor.fetchall()

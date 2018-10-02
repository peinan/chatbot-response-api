import falcon

import json
from json.decoder import JSONDecodeError

from vectorizers.vectorizer import Vectorizer
from evaluators.evaluator import Evaluator

from utils.logger import Logger


class ResourceBase:
    def __init__(self):
        self.logger = Logger(name='doppel').get_logger()
        self.vectorizer = Vectorizer()
        self.evaluator = Evaluator()
        self.debug_mode = False


class ReplyResource(ResourceBase):
    def __init__(self):
        super(ReplyResource, self).__init__()
        self.logger.info('Server Ready.')

    def on_post(self, req, res):
        data = req.stream.read().decode('utf-8')
        self.logger.info(f'Received: {data}')
        try:
            body = json.loads(data)
        except JSONDecodeError:
            self.logger.info('Jsonize failed.')
            raise falcon.HTTPBadRequest('Wrong format')

        uttr = body['utterance']

        topn = body['maxReplies'] if 'maxReplies' in body else 1
        verbose = body['verbose'] if 'verbose' in body else False

        # validation
        if len(uttr.strip()) == 0:
            self.logger.info('Empty utterance field.')
            raise falcon.HTTPBadRequest('"utterance" cannot be empty.')

        if type(topn) != int:
            self.logger.info(f'Wrong topn type: {type(topn)}.')
            raise falcon.HTTPBadRequest('"maxReplies" must be a int value.')

        if topn < 1 or topn > 10:
            self.logger.info(f'Wrong topn range: {topn}.')
            raise falcon.HTTPBadRequest('"maxReplies" must be in the range of [1, 10].')

        if len(uttr.strip()) >= 7 and uttr.endswith('でばっぐだよ☆'):
            self.debug_mode = True
            uttr = uttr.replace('でばっぐだよ☆', '').strip()
            self.logger.info('DEBUG MODE')

        replies = self.__get_replies(uttr, topn, verbose)
        res_body = {'replies': replies}

        res.body = json.dumps(res_body, ensure_ascii=False)
        res.status = falcon.HTTP_200

        self.logger.info(f'RETURN STATUS: {res.status}')
        self.logger.info(f'RETURN BODY: {res.body}')

    def __get_replies(self, utterance: str, topn: int, verbose: bool):
        uttr_vecs = self.vectorizer.vectorize(utterance)
        replies = self.evaluator.get_topn_replies(uttr_vecs, topn)

        if self.debug_mode:
            for reply in replies:
                reply['text'] += '\n\n<DEBUG>\n\n'
                for k, v in reply.items():
                    if k == 'text':
                        continue
                    element = f'{k}: {v}\n'
                    reply['text'] += element

        if not verbose:
            replies = [ {'text': reply['text'], 'similarity': reply['similarity']}
                        for reply in replies ]

        return replies


reply_api = ReplyResource()

app = falcon.API()
app.add_route('/api/replies', reply_api)

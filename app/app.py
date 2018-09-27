import falcon

import json
from json.decoder import JSONDecodeError

from vectorizers.vectorizer import Vectorizer
from evaluators.evaluator import Evaluator


class ResourceBase:
    def __init__(self):
        self.vectorizer = Vectorizer()
        self.evaluator = Evaluator()


class ReplyResource(ResourceBase):
    def __init__(self):
        super(ReplyResource, self).__init__()
        print('Server ready.')

    def on_post(self, req, res):
        data = req.stream.read().decode('utf-8')
        print(f'Recieved: {data}')
        try:
            body = json.loads(data)
        except JSONDecodeError:
            raise falcon.HTTPBadRequest('Wrong format')

        uttr = body['utterance']

        topn = body['maxReplies'] if 'maxReplies' in body else 1
        verbose = body['verbose'] if 'verbose' in body else False

        res_body = {'replies': self.__get_replies(uttr, topn, verbose)}

        res.body = json.dumps(res_body, ensure_ascii=False)
        res.status = falcon.HTTP_200

    def __get_replies(self, utterance, topn, verbose):
        uttr_vecs = self.vectorizer.vectorize(utterance)
        replies = self.evaluator.get_topn_replies(uttr_vecs, topn, verbose=verbose)

        return replies


reply_api = ReplyResource()

app = falcon.API()
app.add_route('/api/replies', reply_api)

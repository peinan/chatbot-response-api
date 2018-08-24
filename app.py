import falcon

import json
from json.decoder import JSONDecodeError

from vectorizers.word2vec import VectorizerW2V
from evaluators.evaluator import Evaluator


class ResourceBase:
    def __init__(self):
        self.vectorizer = VectorizerW2V()
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

        topn = body['maxReplies']
        if not body['maxReplies']:
            topn = 1

        res_body = {'replies': self.__get_replies(uttr, topn)}

        res.body = json.dumps(res_body, ensure_ascii=False)
        res.status = falcon.HTTP_200

    def __get_replies(self, utterance, topn):
        uttr_vec = self.vectorizer.sent2vec(utterance)
        replies = self.evaluator.get_topn_replies(uttr_vec, topn)

        return replies


reply_api = ReplyResource()

app = falcon.API()
app.add_route('/api/replies', reply_api)

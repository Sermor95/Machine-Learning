import json

class Config:

    def __init__(self, dataset, criba, threshold, top_feat):
        self.dataset = dataset
        self.config_id = ''
        self.criba = criba
        self.threshold = threshold
        self.top_feat = top_feat
        self.launchers = 0
        self.avg_accuracy = []

    def toJSON(self):
        return json.dumps(self.__dict__, default=lambda x:x.__dict__)
import json

class Config:

    def __init__(self, dataset, criba, reduction):
        self.dataset = dataset
        self.config_id = ''
        self.criba = criba
        self.reduction = reduction
        self.launchers = 0

    def toJSON(self):
        return json.dumps(self.__dict__, default=lambda x:x.__dict__)
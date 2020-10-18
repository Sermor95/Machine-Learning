import json

class Config:

    def __init__(self, dataset, criba, limit, pearson_base, ohe, categorical_features):
        self.dataset = dataset
        self.criba = criba
        self.limit = limit
        self.pearson_base = pearson_base
        self.ohe = ohe
        self.categorical_features = categorical_features

    def toJSON(self):
        return json.dumps(self.__dict__, default=lambda x:x.__dict__)
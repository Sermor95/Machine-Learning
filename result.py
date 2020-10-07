import json

class Result:     
    
    def __init__(self, name, criba, accuracy, time, features):
        self.name = name
        self.criba = criba
        self.accuracy = accuracy
        self.time = time
        self.features = features
    

    def toJSON(self):
        return json.dumps(self.__dict__)
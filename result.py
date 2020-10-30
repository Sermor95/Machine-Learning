import json

class Result:     
    
    def __init__(self, method, criba, accuracy, time, features):
        self.config_id = ''
        self.launch = None
        self.method = method
        self.criba = criba
        self.accuracy = accuracy
        self.time = time
        self.features = features
    

    def toJSON(self):
        return json.dumps(self.__dict__)
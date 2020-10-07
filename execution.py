import json

class Execution:     
    
    def __init__(self, results):
        self.results = results
    
    def toJSON(self):
        return json.dumps(self.__dict__)
from pymongo import MongoClient
from pprint import pprint
import json

def get_db():
    mongoDB = 'mongodb://192.168.0.16:27017/'
    client = MongoClient(mongoDB,
                    username='root',
                    password='example')
    return client['machine_learning']
def save_feature_selection(document):
    db = get_db()
    posts = db.feature_selection
    try:
        print('s1')
        print('document ->{}'.format(getattr(document, 'dataset')))
        # TODO https://stackoverflow.com/questions/1167398/python-access-class-property-from-string
        feature_selection = find_one(document.dataset, document.criba, document.ohe, document.categorical_features)
        print('s2')
        executions = feature_selection.execution
        print('s3')
        executions.append(document.execution)
        print('s4')
        posts.update_one({"dataset":document['dataset']}, {"$set": {"execution":executions}, "$currentDate": { "lastModified": True }})
        print('Save done')
        pass
    except:
        posts.insert_one(json.loads(document))
        print('Save fail')
        pass
    

def find_one(dataset, criba, ohe, debug):
    print('f1')
    db = get_db()
    print('f2')
    posts = db.feature_selection
    print('f3')
    try:
        print('Find one done') 
        res = posts.find({"dataset": dataset})#, "criba": criba, "ohe": ohe, "debug": debug
        print('res find one ->{}'.format(res))
        pass
    except:
        print('Find one fail')
        pass
    return res 
        

def get_feature_selection():
    db = get_db()
    collection = db['feature_selection']
    entries = collection.find({})
    print('entries ->\n{}'.format(entries))
    return entries
    
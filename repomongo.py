from pymongo import MongoClient
from pprint import pprint
from config import Config
import json

def get_db():
    mongoDB = 'mongodb://192.168.0.16:27017/'
    client = MongoClient(mongoDB,
                    username='root',
                    password='example')
    return client['machine_learning']


def save_feature_selection(feature_selection):
    db = get_db()
    posts_result = db.fs_result
    posts_config = db.fs_config
    try:
        config = find_one_config(feature_selection.dataset, feature_selection.criba, feature_selection.ohe, feature_selection.categorical_features)
        if config:
            fs_id = config['_id']
        else:
            config = Config(feature_selection.dataset, feature_selection.criba, feature_selection.ohe, feature_selection.categorical_features)
            inserted = posts_config.insert_one(json.loads(config.toJSON()))
            fs_id = inserted.inserted_id
        results = []
        for result in feature_selection.results:
            res = json.loads(result)
            res['feature_selection_id'] = fs_id
            res['launch'] = count_result_by_config(fs_id)+1
            results.append(res)
        posts_result.insert_many(results)
    except Exception as e:
        print(f'Save fail: {e}')
        pass


def find_one_config(dataset, criba, ohe, categorical_features):
    db = get_db()
    posts = db.fs_config
    try:
        res = posts.find_one({"dataset": dataset, "criba": criba, "ohe": ohe, "categorical_features": categorical_features})
        pass
    except Exception as e:
        print(f'Save fail: {e}')
    return res

def count_result_by_config(config_id):
    db = get_db()
    posts_result = db.fs_result
    return posts_result.count({ "feature_selection_id": config_id})/6

def get_documents(object):
    table = 'fs_'+object
    db = get_db()
    collection = db[table]
    entries = collection.find({})
    return entries

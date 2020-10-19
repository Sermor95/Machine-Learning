from pymongo import MongoClient
from pprint import pprint
from config import Config
from bson import ObjectId
import json

def get_db():
    mongoDB = 'mongodb://192.168.0.14:27017/'
    client = MongoClient(mongoDB,
                    username='root',
                    password='example')
    return client['machine_learning']

def update_config(key,old_value,new_value):
    db = get_db()
    posts_config = db.fs_config
    posts_config.update_one({key:old_value},{"$set": {key:new_value}})

def save_feature_selection(feature_selection):
    db = get_db()
    posts_result = db.fs_result
    posts_config = db.fs_config
    try:
        config = find_one_config(feature_selection.dataset, feature_selection.criba, feature_selection.limit, feature_selection.pearson_base, feature_selection.ohe, feature_selection.categorical_features)
        if config:
            launchers = config['launchers']+1
            fs_id = config['_id']
            update_config("launchers",config['launchers'],launchers)
        else:
            launchers = 1
            config = Config(feature_selection.dataset, feature_selection.criba, feature_selection.limit, feature_selection.pearson_base, feature_selection.ohe, feature_selection.categorical_features)
            config.launchers = launchers
            inserted = posts_config.insert_one(json.loads(config.toJSON()))
            fs_id = inserted.inserted_id
        results = []
        for result in feature_selection.results:
            res = json.loads(result)
            res['config_id'] = fs_id
            res['launch'] = launchers
            results.append(res)
        posts_result.insert_many(results)
    except Exception as e:
        print(f'Save fail: {e}')
        pass


def find_one_config(dataset, criba, limit, pearson_base, ohe, categorical_features):
    db = get_db()
    posts = db.fs_config
    try:
        res = posts.find_one({"dataset": dataset, "criba": criba, "limit": limit, "pearson_base": pearson_base, "ohe": ohe, "categorical_features": categorical_features})
        pass
    except Exception as e:
        print(f'Find One fail: {e}')
    return res

def count_result_by_config(config_id):
    db = get_db()
    posts_result = db.fs_result
    return posts_result.count({ "config_id": ObjectId(config_id)})/14

def get_documents(object):
    table = 'fs_'+object
    db = get_db()
    collection = db[table]
    entries = collection.find({})
    return entries

def get_results_by_config(config_id):
    db = get_db()
    posts_result = db.fs_result
    # posts_result.find({"feature_selection_id": ObjectId(config_id)} )
    return posts_result.find({"config_id": ObjectId(config_id)})

def get_distinct_from_results(attr):
    db = get_db()
    collection = db['fs_result']
    return collection.find().distinct(attr)

def find_configs_by(dataset):
    db = get_db()
    posts = db.fs_config
    try:
        res = posts.find({"dataset": dataset})
        pass
    except Exception as e:
        print(f'Find CONFIG fail: {e}')
    return res

def find_results_by(config_id,method,criba):
    db = get_db()
    posts = db.fs_result
    try:
        res = posts.find({"config_id": ObjectId(config_id), "method": method, "criba": criba})
        pass
    except Exception as e:
        print(f'Find By method and criba fail: {e}')
    return res
from pymongo import MongoClient
from config import Config
from bson import ObjectId
import json
import logging

def get_db():
    mongoDB = 'mongodb://machine-learning_mongo_1/'
    client = MongoClient(mongoDB,
                    username='root',
                    password='example')
    return client['machine_learning']

def get_documents(object):
    table = 'fs_'+object
    db = get_db()
    collection = db[table]
    entries = collection.find({})
    return entries

def save_feature_selection(feature_selection):
    db = get_db()
    posts_result = db.fs_result
    try:
        config = find_one_config(feature_selection.dataset, feature_selection.criba, feature_selection.reduction, feature_selection.model)
        # Update RESULTS from CONFIG
        if config:
            launchers = config['launchers']+1
            fs_id = config['_id']
            update_config(fs_id,"launchers",launchers)
        # Create a new CONFIG
        else:
            launchers = 1
            config = Config(feature_selection.dataset, feature_selection.criba, feature_selection.reduction, feature_selection.model)
            config.launchers = launchers
            inserted = save_config(config)
            fs_id = inserted.inserted_id
        results = []
        for result in feature_selection.results:
            res = json.loads(result)
            res['config_id'] = fs_id
            res['launch'] = launchers
            results.append(res)
        posts_result.insert_many(results)
        return fs_id
    except Exception as e:
        logging.error(f'//---save_feature_selection()--->{e}---//')
        pass


# CRUDS for CONFIG

def save_config(config):
    config_by_dataset = count_configs_by_dataset(config.dataset)
    config.config_id = config.dataset[0:3]+'_conf_'+str(config_by_dataset+1)
    db = get_db()
    posts_config = db.fs_config
    inserted = posts_config.insert_one(json.loads(config.toJSON()))
    return inserted

def update_config(id,key,new_value):
    db = get_db()
    posts_config = db.fs_config
    posts_config.update_one({"_id":ObjectId(id)},{"$set": {key:new_value}})

def insert_configs(configs):
    db = get_db()
    posts_config = db.fs_config
    ids = posts_config.insert_many(configs)
    return ids


def delete_configs():
    db = get_db()
    posts_config = db.fs_config
    deletes = posts_config.delete_many({})
    return deletes

# Queries for CONFIG
def find_config_by_id(id):
    db = get_db()
    posts = db.fs_config
    try:
        res = posts.find_one({"_id": ObjectId(id)})
        pass
    except Exception as e:
        logging.error(f'//---find_config_by_id()--->{e}---//')
    return res

def find_one_config(dataset, criba, reduction, model):
    db = get_db()
    posts = db.fs_config
    try:
        res = posts.find_one({"dataset": dataset, "criba": criba, "reduction": reduction, "model": model})
        pass
    except Exception as e:
        logging.error(f'//---find_one_config()--->{e}---//')
    return res

# def find_one_config_by_model(dataset, criba, reduction, model):
#     db = get_db()
#     posts = db.fs_config
#     try:
#         res = posts.find_one({"dataset": dataset, "criba": criba, "reduction": reduction, "model": model})
#         pass
#     except Exception as e:
#         logging.error(f'//---find_one_config_by_model()--->{e}---//')
#     return res

def find_cofigs_base(dataset, reduction):
    db = get_db()
    posts = db.fs_config
    try:
        res = posts.find({"dataset": dataset, "reduction": reduction})
        pass
    except Exception as e:
        logging.error(f'//---find_cofigs_base()--->{e}---//')
    return list(res)

def get_configs_by_dataset(dataset):
    db = get_db()
    collection = db['fs_config']
    try:
        res = collection.find({"dataset": dataset})
        pass
    except Exception as e:
        logging.error(f'//---get_configs_by_dataset()--->{e}---//')
    return res

def get_configs_by_dataset_criba(dataset,criba):
    db = get_db()
    collection = db['fs_config']
    try:
        res = collection.find({"dataset": dataset, "criba": criba})
        pass
    except Exception as e:
        logging.error(f'//---get_configs_by_dataset_criba()--->{e}---//')
    return res

def get_configs_by_dataset_reduction(dataset,reduction):
    db = get_db()
    collection = db['fs_config']
    try:
        res = collection.find({"dataset": dataset, "reduction": reduction})
        pass
    except Exception as e:
        logging.error(f'//---get_configs_by_dataset_reduction()--->{e}---//')
    return res

def get_configs_by_dataset_model(dataset,model):
    db = get_db()
    collection = db['fs_config']
    try:
        res = collection.find({"dataset": dataset, "model": model})
        pass
    except Exception as e:
        logging.error(f'//---get_configs_by_dataset_model()--->{e}---//')
    return res

def get_configs_by_dataset_criba_reduction_model(dataset,criba,reduction,model):
    db = get_db()
    collection = db['fs_config']
    try:
        res = collection.find({"dataset": dataset, "criba": criba, "reduction": reduction, "model": model})
        pass
    except Exception as e:
        logging.error(f'//---get_configs_by_dataset_criba_reduction_model()--->{e}---//')
    return res

def count_configs_by_dataset(dataset):
    db = get_db()
    posts_result = db.fs_config
    return posts_result.count({"dataset": dataset})

def get_distinct_from_config(attr):
    db = get_db()
    collection = db['fs_config']
    return collection.find().distinct(attr)


# CRUDS for RESULT

def insert_results(results):
    db = get_db()
    posts_result = db.fs_result
    ids = posts_result.insert_many(results)
    return ids

def delete_results():
    db = get_db()
    documents = db.fs_result
    deletes = documents.delete_many({})
    return deletes

def delete_results_not_base():
    db = get_db()
    documents = db.fs_result
    deletes = documents.delete_many({ "time": None })
    return deletes

# Queries for RESULT

def count_result_by_config(config_id):
    db = get_db()
    posts_result = db.fs_result
    return posts_result.count({ "config_id": ObjectId(config_id)})/18

def get_results_by_configid_method_criba(config_id,method,criba):
    db = get_db()
    posts = db.fs_result
    try:
        res = posts.find({"config_id": ObjectId(config_id), "method": method, "criba": criba})
        pass
    except Exception as e:
        logging.error(f'//---get_results_by_configid_method_criba()--->{e}---//')
    return res

def get_results_by_configid_method_criba(config_id,method,criba):
    db = get_db()
    posts = db.fs_result
    try:
        res = posts.find({"config_id": ObjectId(config_id), "method": method, "criba": criba})
        pass
    except Exception as e:
        logging.error(f'//---count_result_by_config()--->{e}---//')
    return res

def get_results_by_configid_method(config_id,method):
    db = get_db()
    posts = db.fs_result
    try:
        res = posts.find({"config_id": ObjectId(config_id), "method": method})
        pass
    except Exception as e:
        logging.error(f'//---get_results_by_configid_method()--->{e}---//')
    return res

def get_results_by_configid_criba(config_id,criba):
    db = get_db()
    posts = db.fs_result
    try:
        res = posts.find({"config_id": ObjectId(config_id), "criba": criba})
        pass
    except Exception as e:
        logging.error(f'//---get_results_by_configid_criba()--->{e}---//')
    return res


def get_avg_results_by_configid_method_criba(config_id,method,criba):
    results = get_results_by_configid_method_criba(config_id,method,criba);
    maped_list = list(map(lambda res: res['accuracy'], results))
    return sum(maped_list)/len(maped_list)


def get_distinct_from_results(attr):
    db = get_db()
    collection = db['fs_result']
    return collection.find().distinct(attr)

def get_results_by_config(config_id):
    db = get_db()
    posts_result = db.fs_result
    # posts_result.find({"feature_selection_id": ObjectId(config_id)} )
    return posts_result.find({"config_id": ObjectId(config_id)})
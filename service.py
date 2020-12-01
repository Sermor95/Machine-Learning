from myprocessor import FeatureSelection
from repomongo import *

def apply_feature_selection(json_request):
    launchers = json_request['launchers']
    config = find_one_config(json_request['dataset'], json_request['criba'], 0, json_request['model'])

    # Create a base configuration
    if config == None:
        feat_selection_1 = FeatureSelection(json_request['dataset'], json_request['criba'], 0, json_request['model'])
        feat_selection_1.procces_full()
        conf_id = save_feature_selection(feat_selection_1)

    # Crete a configuration with reduction
    else:
        conf_id = config['_id']
    for i in range(launchers):
        feat_selection_2 = FeatureSelection(json_request['dataset'], json_request['criba'], json_request['reduction'],
                                            json_request['model'])
        feat_selection_2.id = conf_id
        feat_selection_2.procces_reduction()
        save_feature_selection(feat_selection_2)
    return get_documents('config')
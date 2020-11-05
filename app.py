from flask import Flask, request, render_template, current_app
from bson.json_util import loads
from myprocessor import FeatureSelection
from functions import *
from repomongo import *
import logging
import traceback

app = Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
    # logging.debug('This is a debug message')
    # logging.info('This is an info message')
    # logging.warning('This is a warning message')
    # logging.error('This is an error message')
    # logging.critical('This is a critical message')
    rows = get_documents('config')
    return render_template("configs.html", entries=rows)

@app.route('/analyze-config', methods=["GET"])
def analyze_config():
    dataset = request.args.get('dataset')
    configs = get_configs_by_dataset(dataset)
    res = get_avg_accuracy_by_configs(configs)
    return res

@app.route('/analyze-result', methods=["GET"])
def analyze_result():
    config_id = request.args.get('config')
    config = []
    config.append(find_config_by_id(config_id))
    res = get_avg_accuracy_by_configs(config)
    return res

@app.route('/launch', methods=["GET"])
def launch():
    rows = get_documents('config')
    return render_template("configs.html", entries=rows)

@app.route('/result', methods=["GET"])
def results():
    try:
        config_id = request.args.get('config').replace('\'','')
        rows = get_results_by_config(config_id)
        config = find_config_by_id(config_id)
        methods = get_distinct_from_results('method')
        return render_template("results.html", entries=rows, methods=methods, config=config)
    except Exception as e:
        # output = str(e)
        logging.error(f'//---results()--->{e}---//')
        traceback.print_exc()
        return error(e,traceback.print_exc())

@app.route('/reset-database', methods=["GET"])
def reset():
    try:
        reset_database()
        return homepage()
    except Exception as e:
        # output = str(e)
        logging.error(f'//---reset()--->{e}---//')
        traceback.print_exc()
        return error(e,traceback.print_exc())

def error(exception, trace):
    return render_template("error.html", exception=exception, trace=trace)

@app.route('/feature-selection', methods=["POST"])
def feature_selection():
    try:
        json_request = request.get_json()
        launchers = json_request['launchers']
        config = find_one_config(json_request['dataset'], json_request['criba'], 0)
        if config == None:
            feat_selection_1 = FeatureSelection(json_request['dataset'], json_request['criba'], 0)
            feat_selection_1.procces_full()
            conf_id = save_feature_selection(feat_selection_1)

        else:
            conf_id = config['_id']
        for i in range(launchers):
            feat_selection_2 = FeatureSelection(json_request['dataset'], json_request['criba'], json_request['reduction'])
            feat_selection_2.id = conf_id
            feat_selection_2.procces_reduction()
            save_feature_selection(feat_selection_2)
        rows = get_documents('config')
        return render_template("configs.html", entries=rows)
    except Exception as e:
        # output = str(e)
        logging.error(f'//---feature_selection()--->{e}---//')
        traceback.print_exc()
        return error(e, traceback.print_exc())


@app.route('/analyze-results', methods=["GET"])
def filter_results():
    config_id = request.args.get('config')
    method = request.args.get('method')
    criba = loads(request.args.get('criba').lower())
    results = get_results_by_configid_method_criba(config_id,method,criba)
    return render_template("results.html", entries=results, method=method, criba=criba)


if __name__ == "__main__":
    app.run()
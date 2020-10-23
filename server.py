from flask import Flask, request, render_template
from bson.json_util import loads
from myprocessor import FeatureSelection
from functions import *
from repomongo import *

app = Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
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
    config_id = request.args.get('config').replace('\'','')
    rows = get_results_by_config(config_id)
    methods = get_distinct_from_results('method')
    return render_template("results.html", entries=rows, methods=methods, config=config_id)

@app.route('/feature-selection', methods=["POST"])
def feature_selection():
    try:
        json_request = request.get_json()
        feat_selection = FeatureSelection(json_request['dataset'], json_request['criba'], json_request['reduction'])
        feat_selection.procesa()
        # output = ''.join(feat_selection.getResultados())
        save_feature_selection(feat_selection)
    except Exception as e:
        # output = str(e)
        print(f'Procces fail: {e}')
    return homepage()

@app.route('/analyze-results', methods=["GET"])
def filter_results():
    config_id = request.args.get('config')
    method = request.args.get('method')
    criba = loads(request.args.get('criba').lower())
    results = get_results_by_configid_method_criba(config_id,method,criba)
    return render_template("results.html", entries=results, method=method, criba=criba)


if __name__ == "__main__":
    app.run()
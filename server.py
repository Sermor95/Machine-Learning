from flask import Flask, request, render_template
import sys
from bson.json_util import dumps
from bson.json_util import loads

from myprocessor import FeatureSelection
from repomongo import *

app = Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
    rows = get_documents('config')
    return render_template("configs.html", entries=rows)

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
    json_request = request.get_json()
    try:
        feat_selection = FeatureSelection(json_request['dataset'], json_request['criba'], json_request['limit'], json_request['pearson_base'], json_request['ohe'], json_request['categorical_features'])
        feat_selection.procesa()
        # output = ''.join(feat_selection.getResultados())
        save_feature_selection(feat_selection)
    except Exception as e:
        # output = str(e)
        print(f'Save fail: {e}')
    return homepage()

@app.route('/analyze', methods=["GET"])
def filter_results():
    config_id = request.args.get('config')
    method = request.args.get('method')
    criba = loads(request.args.get('criba').lower())
    results = find_results_by(config_id,method,criba)
    return render_template("results.html", entries=results, method=method, criba=criba)


if __name__ == "__main__":
    app.run()
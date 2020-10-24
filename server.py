from flask import Flask, request, render_template
import sys
from bson.json_util import dumps
from bson.json_util import loads
from myprocessor import FeatureSelection
from functions import *
from repomongo import *

app = Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
    rows = get_documents('config')
    return render_template("configs.html", entries=rows)


# def get_avg_accuracy_by_configs(configs):
#     methods = get_methods()
#     cwoc = []
#     cwc = []
#     woc = []
#     wc = []
#     for c in configs:
#         cwoc.append(c['config_id'] + '_woc')
#         cwc.append(c['config_id'] + '_wc')
#         woc_aux = []
#         wc_aux = []
#         for m in methods:
#             woc_aux.append(get_avg_results_by_configid_method_criba(c['_id'], m, False)) # aplicar sumatorio
#             wc_aux.append(get_avg_results_by_configid_method_criba(c['_id'], m, True))
#         woc.append(woc_aux)
#         wc.append(woc_aux)
#
#     return 'lets go'


# def get_avg_accuracy_by_configs(configs):
#     methods = get_methods()
#
#     configs_custom_woc = []
#     configs_custom_wc = []
#     for c in configs:
#         results_woc = []
#         results_wc = []
#         for m in methods:
#             avg_woc = get_avg_results_by_configid_method_criba(c['_id'], m, False)
#             results_woc.append(avg_woc)
#
#             avg_wc = get_avg_results_by_configid_method_criba(c['_id'], m, True)
#             results_wc.append(avg_wc)
#
#         config_woc = {
#             'name': c['config_id']+'_woc',
#             'data': results_woc
#         }
#         configs_custom_woc.append(config_woc)
#
#         config_wc = {
#             'name': c['config_id']+'_wc',
#             'data': results_wc
#         }
#         configs_custom_wc.append(config_wc)
#
#
#     categories = methods
#     series = configs_custom_woc+configs_custom_wc
#     res = {
#         'categories': categories,
#         'series': series
#     }
#     return res
def get_avg_accuracy_by_configs(configs):
    methods = get_methods()
    configs_custom_woc = []
    configs_custom_wc = []
    for c in configs:
        results_woc = []
        results_wc = []
        for m in methods:
            avg_woc = get_avg_results_by_configid_method_criba(c['_id'], m, False)
            results_woc.append(avg_woc)

            avg_wc = get_avg_results_by_configid_method_criba(c['_id'], m, True)
            results_wc.append(avg_wc)
        config_woc = []
        config_name = c['config_id']+'_woc'
        config_woc.append(config_name)
        config_woc = config_woc+results_woc
        configs_custom_woc.append(config_woc)

        config_wc = []
        config_name = c['config_id'] + '_wc'
        config_wc.append(config_name)
        config_wc = config_wc + results_wc
        configs_custom_wc.append(config_wc)


    categories = ['methods']
    for m in methods:
        categories.append(m)
    series = [categories]
    for i in range(len(configs_custom_woc)):
        series.append(configs_custom_woc[i])
        series.append(configs_custom_wc[i])

    res = {
        'categories': get_methods(),
        'series': series
    }
    return res

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
        feat_selection = FeatureSelection(json_request['dataset'], json_request['criba'], json_request['top_feat'], json_request['threshold'])
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
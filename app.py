from flask import Flask, request, render_template
from bson.json_util import loads

from service import apply_feature_selection
from util import *
from repomongo import *
import logging
import traceback,sys

app = Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
    rows = get_documents('config')
    datasets = get_distinct_from_config('dataset')
    cribas = get_distinct_from_config('criba')
    reductions = get_distinct_from_config('reduction')
    models = get_distinct_from_config('model')
    attributes = ['criba', 'reduction', 'model']
    return render_template("configs.html", entries=rows, datasets=datasets, cribas=cribas, reductions=reductions, models=models, attributes=attributes)


@app.route('/get_distinct_from_config', methods=["GET"])
def distinct_from_config():
    attr = request.args.get('attr')
    json = {'attr': attr,
            'values': get_distinct_from_config(attr)
        }
    return json

@app.route('/analyze-config', methods=["GET"])
def analyze_config():
    dataset = request.args.get('dataset')
    attribute = request.args.get('attribute')
    value = request.args.get('value')

    is_base = False
    configs = []
    res = []
    times = []

    cribas = get_distinct_from_config('criba')
    cribas.sort(reverse=True)
    reductions = get_distinct_from_config('reduction')
    reductions.sort(reverse=False)
    models = ['decision-tree', 'random-forest', 'gradient-boosting']

    if attribute == 'criba':
        configs = get_configs_by_dataset_criba(dataset,float(value))
        for r in reductions:
            for m in models:
                chart = list(get_configs_by_dataset_criba_reduction_model(dataset,float(value),r,m))
                if len(chart) != 0:
                    res.append({'title': 'Reduction: '+str(r)+'% - Model: '+m,
                                    'chartParam': get_avg_accuracy_by_configs(chart)})
    elif attribute == 'reduction':
        if int(value)==0:
            is_base = True
        configs = get_configs_by_dataset_reduction(dataset, int(value))
        for c in cribas:
            for m in models:
                chart = list(get_configs_by_dataset_criba_reduction_model(dataset,c,int(value),m))
                if len(chart) != 0:
                    res.append({'title': 'Criba: '+str(c)+' - Model: '+m,
                                    'chartParam': get_avg_accuracy_by_configs(chart)})
                    # Si la configuración es base (reduccion = 0) y solo debría haber una base por configuracion =>
                    # añadimos grafica de tiempos
                    if is_base and len(chart)==1:
                        times.append(get_chart_of_times(chart))
    elif attribute == 'model':
        configs = get_configs_by_dataset_model(dataset, value)
        for c in cribas:
            for r in reductions:
                chart = list(get_configs_by_dataset_criba_reduction_model(dataset,c,r,value))
                if len(chart) != 0:
                    res.append({'title': 'Criba: '+str(c)+' - Reduction: '+str(r)+'%',
                                    'chartParam': get_avg_accuracy_by_configs(chart)})

    html = render_template("configs.html", entries=configs)
    json = {
        'res': res,
        'is_base': is_base,
        'times': times,
        'template': html
    }
    return json

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

def error(exception, trace):
    return render_template("error.html", exception=exception, trace=trace)

@app.route('/feature-selection', methods=["POST"])
def feature_selection():
    try:
        json_request = request.get_json()
        rows = apply_feature_selection(json_request)
        return render_template("configs.html", entries=rows)
    except Exception as e:
        exc_type, exc_value, exc_traceback = sys.exc_info()

        # output = str(e)
        logging.error(f'//---feature_selection()--->{e}---//')
        traceback.print_exc()
        return error(e, traceback.extract_tb(exc_traceback))


@app.route('/analyze-results', methods=["GET"])
def filter_results():
    config_id = request.args.get('config')
    try:
        method = request.args.get('method')
    except:
        method = ''
    try:
        criba = loads(request.args.get('criba').lower())
    except:
        criba = ''

    if method != '' and criba != '':
        results = get_results_by_configid_method_criba(config_id,method,criba)
    elif method != '':
        results = get_results_by_configid_method(config_id, method)
    elif criba != '':
        results = get_results_by_configid_criba(config_id, criba)
    else:
        results = None
    return render_template("results.html", entries=results, method=method, criba=criba)



@app.route('/populate', methods=["GET"])
def populate():
    documents = []
    try:
        delete_configs()
        delete_results()
        configs = get_init_config()
        for c in configs:
            res = apply_feature_selection(c)
            documents.append(res)
        return '200 OK'
    except Exception as e:
        return e

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


if __name__ == "__main__":
    app.run()
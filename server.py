from flask import Flask, request, render_template
import sys

from myprocessor import FeatureSelection
from repomongo import *

app = Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
    rows = get_documents('config')
    return render_template("index.html", entries=rows)

@app.route('/result', methods=["GET"])
def results():
    config_id = request.args.get('config')
    rows = get_documents('result')
    return render_template("results.html", entries=rows)


@app.route('/feature-selection', methods=["POST"])
def feature_selection():
    json_request = request.get_json()
    
    try:
        #integer tiempo = myfdetec.procesa()
        feature_selection = FeatureSelection(json_request['dataset'], json_request['criba'], json_request['ohe'], json_request['categorical_features'])
        feature_selection.procesa()
        # print('1----->\n: {}'.format(feature_selection.getResultados()))
        output = ''.join(feature_selection.getResultados())
        # print('2----->\n: {}'.format(output))
        save_feature_selection(feature_selection)
        # print('3----->\n')
    except Exception as e:
        output = str(e)
        print(f'Save fail: {e}')
    return output

if __name__ == "__main__":
    app.run()
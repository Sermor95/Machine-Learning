from flask import Flask, request, render_template
import sys

from myprocessor import FeatureSelection
from repomongo import *

app = Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
    feature_selection = get_feature_selection()
    return render_template("index.html", entries=feature_selection)


@app.route('/feature-selection', methods=["POST"])
def feature_selection():
    json_request = request.get_json()
    
    try:
        #integer tiempo = myfdetec.procesa()
        print('0----->\n')
        feature_selection = FeatureSelection(json_request['dataset'], json_request['criba'], json_request['ohe'], json_request['categorical_features'])
        print('0.1----->\n')
        feature_selection.procesa()
        # print('1----->\n: {}'.format(feature_selection.getResultados()))
        output = ''.join(feature_selection.getResultados())
        # print('2----->\n: {}'.format(output))
        save_feature_selection(feature_selection.toJSON())
        # print('3----->\n')
    except:
        output = 'Bad request'
    return output

if __name__ == "__main__":
    app.run()
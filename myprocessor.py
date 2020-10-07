import json
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from repomongo import *
from functions import *
from execution import Execution


class FeatureSelection:     
    
    def __init__(self, dataset, criba, ohe, categorical_features):
        self.dataset = dataset
        self.criba = criba
        self.ohe = ohe
        self.categorical_features = categorical_features
        self.execution = []
        # self.debug_info = []

    def procesa(self):
        execution = []
        # CARGA DE DATOS
        data = pd.read_csv(self.dataset)
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]
        print('debug CARGA DE DATOS')

        # ONE HOT ENCODING
        if self.categorical_features: # if list is not empty
            features = self.categorical_features
        else:
            features = X.columns
        X = apply_one_hot_encoding(X,features)
        print('debug ONE HOT ENCODING')

        # SPLIT
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print('debug SPLIT')

        # CRIBA
        criba = criba_Pearson(X_train,y_train,0.75,'pearson')
        features_to_drop = criba[0]
        
        X_train_new = X_train.drop(features_to_drop,axis=1)

        features_without_criba = [list(X_train.columns),2000]
        features_with_criba = [list(list(set(X_train.columns) - set(features_to_drop))),2000]
        execution.append(get_result('Criba Person', features_without_criba, False, X_train, X_test, y_train, y_test))
        execution.append(get_result('Criba Person', features_with_criba, True, X_train, X_test, y_train, y_test))

        print('debug CRIBA')

        # FILTRO: PEARSON CORRELATION
        best_features_pearson_s = feature_selection('pearson',X_train, y_train,0.0)
        best_features_pearson_c = feature_selection('pearson',X_train_new, y_train,0.0)
        execution.append(get_result('Person Correlation', best_features_pearson_s, False, X_train, X_test, y_train, y_test))
        execution.append(get_result('Person Correlation', best_features_pearson_c, True, X_train, X_test, y_train, y_test))
        print('debug PEARSON CORRELATION')

        # ENVOLTURA: FORWARD SELECTION
        # best_features_forward_s = feature_selection('forward',X_train, y_train,20)
        # best_features_forward_c = feature_selection('forward',X_train_new,y_train,20)
        # self.execution.append(get_result('Forward Selection', best_features_forward_s, False, X_train, X_test, y_train, y_test))
        # self.execution.append(get_result('Forward Selection', best_features_forward_c, True, X_train, X_test, y_train, y_test))

        print('debug FORWARD SELECTION')
        # INTEGRADO: FEATURE IMPORTANCE
        best_features_importance_s = feature_selection('feature_importance',X_train, y_train,20)
        best_features_importance_c = feature_selection('feature_importance',X_train_new,y_train,20)
        execution.append(get_result('Feature Importance', best_features_importance_s, False, X_train, X_test, y_train, y_test))
        execution.append(get_result('Feature Importance', best_features_importance_c, True, X_train, X_test, y_train, y_test))
        print('debug FEATURE IMPORTANCE')
        self.execution = execution

    def getResultados(self):
        return self.execution
    
    def toJSON(self):
        return json.dumps(self.__dict__)
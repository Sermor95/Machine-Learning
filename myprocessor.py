import json
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from repomongo import *
from functions import *


class FeatureSelection:     
    
    def __init__(self, dataset, criba, limit, pearson_base, ohe, categorical_features):
        self.dataset = dataset
        self.criba = criba
        self.limit = limit
        self.pearson_base = pearson_base
        self.ohe = ohe
        self.categorical_features = categorical_features
        self.results = []
        # self.debug_info = []

    def procesa(self):
        limit = self.limit
        pearson_base = self.pearson_base
        results = []
        # CARGA DE DATOS
        data = pd.read_csv(self.dataset)
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]

        # ONE HOT ENCODING
        if self.categorical_features: # if list is not empty
            features = self.categorical_features
        else:
            features = X.columns
        X = apply_one_hot_encoding(X,features)

        # SPLIT
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # CRIBA
        criba = criba_Pearson(X_train,y_train,0.75,'pearson')
        features_to_drop = criba[0]
        
        X_train_new = X_train.drop(features_to_drop,axis=1)

        features_without_criba = [list(X_train.columns),criba[1]]
        features_with_criba = [list(list(set(X_train.columns) - set(features_to_drop))),criba[1]]
        results.append(get_result('Criba Person', features_without_criba, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Criba Person', features_with_criba, True, X_train, X_test, y_train, y_test))

        # FILTRO: PEARSON CORRELATION
        best_features_pearson_s = feature_selection('pearson',X_train, y_train,limit,pearson_base)
        best_features_pearson_c = feature_selection('pearson',X_train_new, y_train,limit,pearson_base)
        results.append(get_result('Person Correlation', best_features_pearson_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Person Correlation', best_features_pearson_c, True, X_train, X_test, y_train, y_test))
        print('pearson')

        # ENVOLTURA: FORWARD SELECTION
        best_features_forward_s = feature_selection('forward',X_train, y_train,limit,None)
        best_features_forward_c = feature_selection('forward',X_train_new,y_train,limit,None)
        results.append(get_result('Forward Selection', best_features_forward_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Forward Selection', best_features_forward_c, True, X_train, X_test, y_train, y_test))
        print('forward')

        # ENVOLTURA: BACKWARD SELECTION
        best_features_forward_s = feature_selection('backward', X_train, y_train, limit,None)
        best_features_forward_c = feature_selection('backward', X_train_new, y_train, limit,None)
        results.append(get_result('Backward Selection', best_features_forward_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Backward Selection', best_features_forward_c, True, X_train, X_test, y_train, y_test))
        print('backward')

        # ENVOLTURA: FORWARD FLOATING SELECTION
        best_features_forward_s = feature_selection('forward_floating', X_train, y_train, limit,None)
        best_features_forward_c = feature_selection('forward_floating', X_train_new, y_train, limit,None)
        results.append(get_result('Forward Floating Selection', best_features_forward_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Forward Floating Selection', best_features_forward_c, True, X_train, X_test, y_train, y_test))
        print('forward_floating')

        # ENVOLTURA: BACKWARD FLOATING SELECTION
        best_features_forward_s = feature_selection('backward_floating', X_train, y_train, limit,None)
        best_features_forward_c = feature_selection('backward_floating', X_train_new, y_train, limit,None)
        results.append(get_result('Backward Floating Selection', best_features_forward_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Backward Floating Selection', best_features_forward_c, True, X_train, X_test, y_train, y_test))
        print('backward_floating')

        # INTEGRADO: FEATURE IMPORTANCE
        best_features_importance_s = feature_selection('feature_importance',X_train, y_train,limit,None)
        best_features_importance_c = feature_selection('feature_importance',X_train_new,y_train,limit,None)
        results.append(get_result('Feature Importance', best_features_importance_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Feature Importance', best_features_importance_c, True, X_train, X_test, y_train, y_test))
        print('feature_importance')

        self.results = results

    def getResultados(self):
        return self.results
    
    def toJSON(self):
        return json.dumps(self.__dict__)
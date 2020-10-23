import json
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from repomongo import *
from functions import *



class FeatureSelection:
    
    def __init__(self, dataset, criba, reduction):
        self.dataset = dataset
        self.criba = criba
        self.reduction = reduction
        self.results = []
        # self.debug_info = []




    def procesa(self):

        results = []
        # CARGA DE DATOS
        url='C:/Users/sergi/Google Drive/Datasets/'+self.dataset+'.csv'
        data = pd.read_csv(url)
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]

        top_feat = get_top_feat(X.num_columns, self.reduction)

        # ONE HOT ENCODING
        if need_ohe(self.dataset, X):
            X = apply_one_hot_encoding(X)

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

        # FILTER: PEARSON CORRELATION
        best_features_pearson_s = feature_selection('pearson',X_train, y_train,top_feat)
        best_features_pearson_c = feature_selection('pearson',X_train_new, y_train,top_feat)
        results.append(get_result('Person Correlation', best_features_pearson_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Person Correlation', best_features_pearson_c, True, X_train, X_test, y_train, y_test))
        print('1/8')

        # FILTER: MUTUAL INFORMATION
        best_features_mutual_s = feature_selection('mutual_information', X_train, y_train, top_feat)
        best_features_mutual_c = feature_selection('mutual_information', X_train_new, y_train, top_feat)
        results.append(
            get_result('Mutual Information', best_features_mutual_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Mutual Information', best_features_mutual_c, True, X_train, X_test, y_train, y_test))
        print('2/8')

        # WRAPPER: FORWARD SELECTION
        best_features_forward_s = feature_selection('forward',X_train, y_train,top_feat)
        best_features_forward_c = feature_selection('forward',X_train_new,y_train,top_feat)
        results.append(get_result('Forward Selection', best_features_forward_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Forward Selection', best_features_forward_c, True, X_train, X_test, y_train, y_test))
        print('3/8')

        # WRAPPER: BACKWARD SELECTION
        best_features_backward_s = feature_selection('backward', X_train, y_train, top_feat)
        best_features_backward_c = feature_selection('backward', X_train_new, y_train, top_feat)
        results.append(get_result('Backward Selection', best_features_backward_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Backward Selection', best_features_backward_c, True, X_train, X_test, y_train, y_test))
        print('4/8')

        # WRAPPER: FORWARD FLOATING SELECTION
        best_features_forward_float_s = feature_selection('forward_floating', X_train, y_train, top_feat)
        best_features_forward_float_c = feature_selection('forward_floating', X_train_new, y_train, top_feat)
        results.append(get_result('Forward Floating Selection', best_features_forward_float_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Forward Floating Selection', best_features_forward_float_c, True, X_train, X_test, y_train, y_test))
        print('5/8')

        # WRAPPER: BACKWARD FLOATING SELECTION
        best_features_forward_float_s = feature_selection('backward_floating', X_train, y_train, top_feat)
        best_features_forward_float_c = feature_selection('backward_floating', X_train_new, y_train, top_feat)
        results.append(get_result('Backward Floating Selection', best_features_forward_float_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Backward Floating Selection', best_features_forward_float_c, True, X_train, X_test, y_train, y_test))
        print('6/8')

        # EMBEDDED: FEATURE IMPORTANCE
        best_features_importance_s = feature_selection('feature_importance',X_train, y_train,top_feat)
        best_features_importance_c = feature_selection('feature_importance',X_train_new,y_train,top_feat)
        results.append(get_result('Feature Importance', best_features_importance_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('Feature Importance', best_features_importance_c, True, X_train, X_test, y_train, y_test))
        print('7/8')

        # HYBRID: RFE
        best_features_rfe_s = feature_selection('RFE', X_train, y_train, top_feat)
        best_features_rfe_c = feature_selection('RFE', X_train_new, y_train, top_feat)
        results.append(
            get_result('RFE', best_features_rfe_s, False, X_train, X_test, y_train, y_test))
        results.append(get_result('RFE', best_features_rfe_c, True, X_train, X_test, y_train, y_test))
        print('8/8')

        self.results = results

    def getResultados(self):
        return self.results
    
    def toJSON(self):
        return json.dumps(self.__dict__)
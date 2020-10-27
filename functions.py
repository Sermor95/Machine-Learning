import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_selection import mutual_info_classif, RFE
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score
from result import Result
import time
from datetime import timedelta
from repomongo import *

def get_methods():
    return ['Criba Person', 'Person Correlation', 'Mutual Information', 'Forward Selection', 'Backward Selection', 'Forward Floating Selection', 'Backward Floating Selection', 'Feature Importance', 'RFE']

def pearson_corr(x, y):
    return abs(round(x.corr(y),6))

def get_worst_feature(feature_i1,feature_i2,X_train,y_train,method_name):
    feature_y = y_train
    feature_1 = X_train.iloc[:,feature_i1]
    feature_2 = X_train.iloc[:,feature_i2]   
    corr1 = pearson_corr(feature_1,feature_y)
    corr2 = pearson_corr(feature_2,feature_y)
    return feature_i1 if corr1 < corr2 else feature_i2

def criba_Pearson(X,y,criba,method_name):
    start_time = time.monotonic()
    res = []
    # debug_info = []
    # debug_info.append('CRIBA PEARSON')
    droped_columns = set()
    corr = X.corr(method=method_name).abs()
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= criba:
                worst_feature = get_worst_feature(i,j,X,y,method_name)
                droped_columns.add(X.columns[worst_feature])
                # debug_info.append(' - {0}/{1} => {2} //--// {0}/y => {3} // {1}/y => {4} //--// worst feature => {5}'.format(X.columns[i],X.columns[j],round(corr.iloc[i,j],4),pearson_corr(X.iloc[:,i],y),pearson_corr(X.iloc[:,j],y),X.columns[worst_feature]))

    X.drop(droped_columns, axis=1)
    res.append(X.columns)
    end_time = time.monotonic()
    res.append(get_execution_time(start_time, end_time))
    # res.append(debug_info)
    return res

def need_ohe(dataset, X):
    if dataset == 'titanic':
        return True
    else:
        return False

def apply_one_hot_encoding(X):
    cat_feat_titanic = ['Embarked', 'Initial', 'Deck', 'Title']
    for col in cat_feat_titanic:
        ohe_col = pd.get_dummies(X[col], prefix=col)
        X = pd.concat([X,ohe_col], axis=1)
        X = X.drop([col], axis=1) 
    return X

def filter_pearson_correlation(X,y,threshold):
    best_features = set()

    for i in range(len(X.columns)):
        corr = pearson_corr(X.iloc[:,i],y)
        if corr > threshold:
            col = X.columns[i]
            best_features.add(col)

    return best_features

def filter_mutual_info_select(X,y,top_feat):
    mi = list(enumerate(mutual_info_classif(X,y)))
    mi.sort(reverse=True, key = lambda x: x[1])
    f_best = list(map(lambda e: e[0], mi))
    # for (ind,rank) in mi:
    #     if rank > threshold:
    #         f_best.append(ind)
    return f_best[0:top_feat]

def wrapper_forward_selection(X,y,n):
    model_forward=sfs(RandomForestRegressor(),k_features=n,forward=True,floating=False,verbose=0,cv=5,n_jobs=-1,scoring='r2')
    model_forward.fit(X,y)
    return list(model_forward.k_feature_names_)

def wrapper_backward_selection(X,y,n):
    model_forward=sfs(RandomForestRegressor(),k_features=n,forward=False,floating=False,verbose=0,cv=5,n_jobs=-1,scoring='r2')
    model_forward.fit(X,y)
    return list(model_forward.k_feature_names_)

def wrapper_forward_floating_selection(X,y,n):
    model_forward=sfs(RandomForestRegressor(),k_features=n,forward=True,floating=True,verbose=0,cv=5,n_jobs=-1,scoring='r2')
    model_forward.fit(X,y)
    return list(model_forward.k_feature_names_)

def wrapper_backward_floating_selection(X,y,n):
    model_forward=sfs(RandomForestRegressor(),k_features=n,forward=False,floating=True,verbose=0,cv=5,n_jobs=-1,scoring='r2')
    model_forward.fit(X,y)
    return list(model_forward.k_feature_names_)

def embedded_feature_importance(X,y,threshold):
    model = RandomForestRegressor()
    model.fit(X, y)
    feat_importance = pd.DataFrame(model.feature_importances_, columns=['Feature_Importance'],
                            index=X.columns)
    feat_importance.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
    best_features = feat_importance[feat_importance['Feature_Importance']>threshold]
    return best_features.index

def hybrid_RFE(X,y,n):
    rfe = RFE(estimator=RandomForestRegressor(), step=1, n_features_to_select=n)
    rfe.fit(X, y)
    return X.columns[np.where(rfe.support_ == True)[0]]

def feature_selection(method,X,y,n,treshold):
    res = []
    start_time = time.monotonic()
    if method == 'pearson':
        res.append(filter_pearson_correlation(X,y,treshold))
    if method == 'forward':
        res.append(wrapper_forward_selection(X,y,n))
    if method == 'backward':
        res.append(wrapper_backward_selection(X,y,n))
    if method == 'forward_floating':
        res.append(wrapper_forward_floating_selection(X,y,n))
    if method == 'backward_floating':
        res.append(wrapper_backward_floating_selection(X,y,n))
    if method == 'feature_importance':
        res.append(embedded_feature_importance(X,y,treshold))
    if method == 'mutual_information':
        res.append(filter_mutual_info_select(X,y,treshold))
    if method == 'RFE':
        res.append(hybrid_RFE(X,y,n))
    end_time = time.monotonic()
    res.append(res.append(get_execution_time(start_time, end_time)))
    return res

def get_result(method_name, feature_selection, criba, X_train, X_test, y_train, y_test):

    X_train_aux = X_train[feature_selection[0]]
    X_test_aux = X_test[feature_selection[0]]
    
    model_without_criba = tree.DecisionTreeClassifier()
    model_without_criba.fit(X_train_aux, y_train)
    y_pred_aux = model_without_criba.predict(X_test_aux)
    bal_accur = balanced_accuracy_score(y_pred_aux, y_test)
    result = Result(method_name, criba, bal_accur, feature_selection[1], list(feature_selection[0])).toJSON()
    return result

def get_execution_time(start_time, end_time):
    diff = timedelta(seconds=end_time - start_time)
    return diff.total_seconds()

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


def get_top_feat(num_columns, reduction):
    return (num_columns*reduction)/100
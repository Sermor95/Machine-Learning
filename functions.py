import pandas as pd
import numpy as np
from sklearn import tree, ensemble
from sklearn.feature_selection import mutual_info_classif, RFE
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.metrics import balanced_accuracy_score
from result import Result
import time
from datetime import timedelta
from repomongo import *
import math



# sagrado

# list_columns = list(map(lambda col: upper[col][upper[col] > 0.8].index.values, corr.columns))
# aux_list = [x for x in range(len(list_columns)) if len(list_columns[x])>0]
# res = [[X.columns[i], list(list_columns[i])] for i in aux_list]
# res_aux = [[sublist[0],element,get_corr(sublist[0],element)] for sublist in res for element in sublist[1]]
# res_aux_1 = sorted(res_aux, key=lambda e: e[2], reverse=True)
# res_aux_1

def top_corr(X, criba):
    get_corr = lambda a, b: corr.iloc[X.columns.get_loc(a), X.columns.get_loc(b)]
    corr = X.corr(method='pearson').abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    list_columns = list(map(lambda col: upper[col][upper[col] > criba].index.values, corr.columns))
    aux_list = [x for x in range(len(list_columns)) if len(list_columns[x]) > 0]
    res = [[X.columns[i], list(list_columns[i])] for i in aux_list]
    res_aux = [[sublist[0], element, get_corr(sublist[0], element)] for sublist in res for element in sublist[1]]
    res_aux_1 = sorted(res_aux, key=lambda e: e[2], reverse=True)
    if res_aux_1:
        return res_aux_1[0]
    else:
        return []

def criba_Pearson(X,y,criba):
    start_time = time.monotonic()
    res = []
    top = top_corr(X, criba)
    while top:
        worst = get_worst_feature(X.columns.get_loc(top[0]), X.columns.get_loc(top[1]), X, y)
        # info.append({'redundancia': top, 'worst': X.columns[worst]})
        X = X.drop([X.columns[worst]], axis=1)
        top = top_corr(X, criba)
    res.append(list(X.columns))
    end_time = time.monotonic()
    res.append(get_execution_time(start_time, end_time))
    # res.append(debug_info)
    return res

def filter_pearson_correlation(X,y,top_feat):
    best_features = {}
    for i in range(len(X.columns)):
        corr = pearson_corr(X.iloc[:,i],y)
        corr = 0 if math.isnan(corr) else corr
        col = X.columns[i]
        best_features[col] = corr
    best_features = sorted(best_features.items(), key=lambda x: x[1], reverse=True)
    res = list(map(lambda k: k[0], best_features))
    return res[:int(top_feat)]

def filter_mutual_info_select(X,y,top_feat):
    mi = list(enumerate(mutual_info_classif(X,y)))
    mi.sort(reverse=True, key=lambda x: x[1])
    f_best = list(map(lambda e: e[0], mi))
    return list(X.columns[f_best[:int(top_feat)]])

def wrapper_forward_selection(X,y,top_feat,model):
    model_forward=sfs(model,k_features=top_feat,forward=True,floating=False,verbose=0,cv=5,n_jobs=-1,scoring='accuracy')
    model_forward.fit(X,y)
    res = list(map(lambda e: e['feature_names'], model_forward.subsets_.values()))
    res.sort(key=len)
    return res

def wrapper_backward_selection(X,y,top_feat,model):
    model_forward=sfs(model,k_features=top_feat,forward=False,floating=False,verbose=0,cv=5,n_jobs=-1,scoring='accuracy')
    model_forward.fit(X,y)
    res = list(map(lambda e: e['feature_names'], model_forward.subsets_.values()))
    res.sort(key=len)
    return res

def wrapper_forward_floating_selection(X,y,top_feat,model):
    model_forward=sfs(model,k_features=top_feat,forward=True,floating=True,verbose=0,cv=5,n_jobs=-1,scoring='accuracy')
    model_forward.fit(X,y)
    res = list(map(lambda e: e['feature_names'], model_forward.subsets_.values()))
    res.sort(key=len)
    return res

def wrapper_backward_floating_selection(X,y,top_feat,model):
    model_forward=sfs(model,k_features=top_feat,forward=False,floating=True,verbose=0,cv=5,n_jobs=-1,scoring='accuracy')
    model_forward.fit(X,y)
    res = list(map(lambda e: e['feature_names'], model_forward.subsets_.values()))
    res.sort(key=len)
    return res

def embedded_feature_importance(X,y,n,model):
    model.fit(X, y)
    feat_importance = pd.DataFrame(model.feature_importances_, columns=['Feature_Importance'],
                            index=X.columns)
    feat_importance.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
    return list(feat_importance.index)[:n]

def hybrid_RFE(X,y,n,model):
    rfe = RFE(estimator=model, step=1, n_features_to_select=1)
    rfe.fit(X, y)
    ranking = pd.DataFrame(rfe.ranking_, columns=['Ranking'], index=X.columns)
    ranking.sort_values(by=['Ranking'], ascending=True, inplace=True)
    return list(ranking.index)[:n]

def feature_selection(method,X,y,n,model):
    res = []
    start_time = time.monotonic()
    if method == 'pearson':
        res.append(filter_pearson_correlation(X,y,n))
    if method == 'mutual_information':
        res.append(filter_mutual_info_select(X, y, n))
    if method == 'forward':
        res.append(wrapper_forward_selection(X,y,n,model))
    if method == 'backward':
        res.append(wrapper_backward_selection(X,y,n,model))
    if method == 'forward_floating':
        res.append(wrapper_forward_floating_selection(X,y,n,model))
    if method == 'backward_floating':
        res.append(wrapper_backward_floating_selection(X,y,n,model))
    if method == 'feature_importance':
        res.append(embedded_feature_importance(X,y,n,model))
    if method == 'RFE':
        res.append(hybrid_RFE(X,y,n,model))
    end_time = time.monotonic()
    res.append(get_execution_time(start_time, end_time))
    return res

def procces_results(features, X_train, X_test, y_train, y_test):


    results = []
    results.append(get_result('Criba Person', features['features_without_criba'], False, X_train, X_test, y_train, y_test))
    results.append(get_result('Criba Person', features['features_with_criba'], True, X_train, X_test, y_train, y_test))

    # FILTER: PEARSON CORRELATION
    results.append(
        get_result('Person Correlation', features['features_pearson_woc'], False, X_train, X_test, y_train, y_test))
    results.append(
        get_result('Person Correlation', features['features_pearson_wc'], True, X_train, X_test, y_train, y_test))


    # FILTER: MUTUAL INFORMATION
    results.append(
        get_result('Mutual Information', features['features_mutual_woc'], False, X_train, X_test, y_train, y_test))
    results.append(
        get_result('Mutual Information', features['features_mutual_wc'], True, X_train, X_test, y_train, y_test))


    # WRAPPER: FORWARD SELECTION
    results.append(
        get_result('Forward Selection', features['features_forward_woc'], False, X_train, X_test, y_train, y_test))
    results.append(
        get_result('Forward Selection', features['features_forward_wc'], True, X_train, X_test, y_train, y_test))


    # WRAPPER: BACKWARD SELECTION
    results.append(
        get_result('Backward Selection', features['features_backward_woc'], False, X_train, X_test, y_train, y_test))
    results.append(
        get_result('Backward Selection', features['features_backward_wc'], True, X_train, X_test, y_train, y_test))


    # WRAPPER: FORWARD FLOATING SELECTION
    results.append(
        get_result('Forward Floating Selection', features['features_forward_float_woc'], False, X_train, X_test, y_train,
                   y_test))

    results.append(
        get_result('Forward Floating Selection', features['features_forward_float_wc'], True, X_train, X_test, y_train,
                   y_test))


    # WRAPPER: BACKWARD FLOATING SELECTION
    results.append(
        get_result('Backward Floating Selection', features['features_forward_float_woc'], False, X_train, X_test, y_train,
                   y_test))
    results.append(
        get_result('Backward Floating Selection', features['features_forward_float_wc'], True, X_train, X_test, y_train,
                   y_test))


    # EMBEDDED: FEATURE IMPORTANCE
    results.append(
        get_result('Feature Importance', features['features_importance_woc'], False, X_train, X_test, y_train, y_test))
    results.append(
        get_result('Feature Importance', features['features_importance_wc'], True, X_train, X_test, y_train, y_test))


    # HYBRID: RFE
    results.append(get_result('RFE', features['features_rfe_woc'], False, X_train, X_test, y_train, y_test))
    results.append(get_result('RFE', features['features_rfe_wc'], True, X_train, X_test, y_train, y_test))


    return results

def get_result(method_name, feature_selection, criba, X_train, X_test, y_train, y_test):

    if any(isinstance(x, tuple) and len(x) > 1 for x in feature_selection[0]):
        X_train_aux = X_train
        X_test_aux = X_test
    else:
        X_train_aux = X_train[feature_selection[0]]
        X_test_aux = X_test[feature_selection[0]]

    model = tree.DecisionTreeClassifier()
    model.fit(X_train_aux, y_train)
    y_pred_aux = model.predict(X_test_aux)
    bal_accur = balanced_accuracy_score(y_pred_aux, y_test)
    if feature_selection[1] == None:
        result = Result(method_name, criba, bal_accur, feature_selection[1], len(feature_selection[0])).toJSON()
    else:
        result = Result(method_name, criba, bal_accur, feature_selection[1], feature_selection[0]).toJSON()
    return result

def get_model(model):
    if model == 'decision-tree':
        return tree.DecisionTreeClassifier()
    elif model == 'random-forest':
        return ensemble.RandomForestClassifier()
    elif model == 'gradient-boosting':
        return ensemble.GradientBoostingClassifier()
def get_datasets():
    return ['titanic', 'BreastCancerDataset', 'spambase']
def get_methods():
    return ['Criba Person', 'Person Correlation', 'Mutual Information', 'Forward Selection', 'Backward Selection', 'Forward Floating Selection', 'Backward Floating Selection', 'Feature Importance', 'RFE']

def need_ohe(dataset, X):
    if dataset == 'titanic':
        return True
    else:
        return False

def apply_one_hot_encoding(X):
    cat_feat_titanic = ['Embarked', 'Initial', 'Deck', 'Title']
    # cat_feat_titanic = []
    for col in cat_feat_titanic:
        ohe_col = pd.get_dummies(X[col], prefix=col)
        X = pd.concat([X,ohe_col], axis=1)
        X = X.drop([col], axis=1)
    return X

def pearson_corr(x, y):
    return abs(round(x.corr(y, method='pearson'),6))

def get_worst_feature(feature_i1,feature_i2,X_train,y_train):
    feature_y = y_train
    feature_1 = X_train.iloc[:,feature_i1]
    feature_2 = X_train.iloc[:,feature_i2]
    corr1 = pearson_corr(feature_1,feature_y)
    corr2 = pearson_corr(feature_2,feature_y)
    return feature_i1 if corr1 < corr2 else feature_i2

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


def get_chart_of_times(config):
    res = []
    methods = get_methods()
    results = list(get_results_by_config(config[0]['_id']))
    for r in results:
        res.append({'name': r['method'],'y': r['time']})
    return res

def get_top_feat(num_columns, reduction):
    return int(num_columns-((num_columns*reduction)/100))

def get_top_feat_by_config(config_id,method,criba,n,is_wraper):
    if method == 'Criba Pearson':
        return list(get_results_by_configid_method_criba(config_id, method, criba))[0]['features']
    elif is_wraper:
        return list(get_results_by_configid_method_criba(config_id, method, criba))[0]['features'][n-1]
    else:
        return list(get_results_by_configid_method_criba(config_id, method, criba))[0]['features'][:n]


def get_top_feat_by_config_sequential(config_id,method,criba,n):
    return list(get_results_by_configid_method_criba(config_id,method,criba))[n-1]



def reset_database():
    configs = []
    for dataset in get_datasets():
        configs += find_cofigs_base(dataset, 0)

    delete_configs()
    delete_results_not_base()
    insert_configs(configs)
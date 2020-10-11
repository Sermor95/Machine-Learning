import pandas as pd
import numpy as numpy
from sklearn import tree
from sklearn.model_selection import train_test_split
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import balanced_accuracy_score
from result import Result
import time
from datetime import timedelta


def pearson_corr(x, y):
    return abs(round(x.corr(y),6))

def get_worst_feature(feature_i1,feature_i2,X_train,y_train,method_name):
    feature_y = y_train
    feature_1 = X_train.iloc[:,feature_i1]
    feature_2 = X_train.iloc[:,feature_i2]   
    corr1 = pearson_corr(feature_1,feature_y)
    corr2 = pearson_corr(feature_2,feature_y)
    return feature_i1 if corr1 < corr2 else feature_i2

def criba_Pearson(X,y,limit,method_name):
    start_time = time.monotonic()
    res = []
    # debug_info = []
    # debug_info.append('CRIBA PEARSON')
    droped_columns = set()
    corr = X.corr(method=method_name).abs()
    for i in range(corr.shape[0]):
        for j in range(i+1, corr.shape[0]):
            if corr.iloc[i,j] >= limit:
                worst_feature = get_worst_feature(i,j,X,y,method_name)
                droped_columns.add(X.columns[worst_feature])
                # debug_info.append(' - {0}/{1} => {2} //--// {0}/y => {3} // {1}/y => {4} //--// worst feature => {5}'.format(X.columns[i],X.columns[j],round(corr.iloc[i,j],4),pearson_corr(X.iloc[:,i],y),pearson_corr(X.iloc[:,j],y),X.columns[worst_feature]))
    
    res.append(droped_columns)
    end_time = time.monotonic()
    res.append(get_execution_time(start_time, end_time))
    # res.append(debug_info)
    return res

def apply_one_hot_encoding(X,categorical_features):
    for col in categorical_features:
        ohe_col = pd.get_dummies(X[col], prefix=col)
        X = pd.concat([X,ohe_col], axis=1)
        X = X.drop([col], axis=1) 
    return X

def pearson_correlation_filter(X,y,base):
    best_features = set()
    for i in range(len(X.columns)):
        corr = pearson_corr(X.iloc[:,i],y)
        if corr > base:
            col = X.columns[i]
            best_features.add(col)
    return best_features

def forward_selection_wrapper(X,y,n):
    model_forward=sfs(RandomForestRegressor(),k_features=n,forward=True,verbose=0,cv=5,n_jobs=-1,scoring='r2')
    model_forward.fit(X,y)
    return list(model_forward.k_feature_names_)

def forward_selection_embedded(X,y,n):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    feat_importance = pd.DataFrame(model.feature_importances_, columns=['Feature_Importance'],
                            index=X.columns)
    feat_importance.sort_values(by=['Feature_Importance'], ascending=False, inplace=True)
    best_features = feat_importance[feat_importance['Feature_Importance']>0] 
    return best_features.index[0:n]

def feature_selection(method,X,y,n):
    res = []
    start_time = time.monotonic()
    if method == 'pearson':
        res.append(pearson_correlation_filter(X,y,n))
    if method == 'forward':
        res.append(forward_selection_wrapper(X,y,n))
    if method == 'feature_importance':
        res.append(forward_selection_embedded(X,y,n))
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
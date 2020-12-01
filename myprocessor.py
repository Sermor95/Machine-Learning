from sklearn.model_selection import train_test_split
from util import *

class FeatureSelection:
    
    def __init__(self, dataset, criba, reduction, model):
        self.id = None
        self.dataset = dataset
        self.criba = criba
        self.reduction = reduction
        self.model = model
        self.results = []
        # self.debug_info = []


    def procces_full(self):
        model = get_model(self.model)
        # CARGA DE DATOS
        url='https://raw.githubusercontent.com/Sermor95/Machine-Learning/master/datasets/'+self.dataset+'.csv'
        data = pd.read_csv(url)
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]

        # ONE HOT ENCODING
        if need_ohe(self.dataset, X):
            X = apply_one_hot_encoding(X)

        # SPLIT
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # CRIBA
        criba = criba_Pearson(X_train,y_train,self.criba)
        X_train_new = X_train[criba[0]]

        top_feat_woc = get_top_feat(len(X.columns), self.reduction)
        top_feat_wc = get_top_feat(len(X_train_new.columns), self.reduction)

        logging_info(f'procces_full()---top_feat_woc', top_feat_woc)
        logging_info(f'procces_full()---top_feat_wc', top_feat_wc)
        features_without_criba = [list(X_train.columns),0.0]
        features_with_criba = [criba[0],criba[1]]

        # FILTER: PEARSON CORRELATION
        best_features_pearson_woc = feature_selection('pearson',X_train, y_train,top_feat_woc,None)
        best_features_pearson_wc = feature_selection('pearson',X_train_new, y_train,top_feat_wc,None)
        logging_info('procces_full()','1/8')

        # FILTER: MUTUAL INFORMATION
        best_features_mutual_woc = feature_selection('mutual_information', X_train, y_train, top_feat_woc,None)
        best_features_mutual_wc = feature_selection('mutual_information', X_train_new, y_train, top_feat_wc,None)
        logging_info('procces_full()','2/8')

        # WRAPPER: FORWARD SELECTION
        best_features_forward_woc = feature_selection('forward',X_train, y_train,top_feat_woc,model)
        best_features_forward_wc = feature_selection('forward',X_train_new,y_train,top_feat_wc,model)
        logging_info('procces_full()','3/8')

        # WRAPPER: BACKWARD SELECTION
        best_features_backward_woc = feature_selection('backward', X_train, y_train, 1,model)
        best_features_backward_wc = feature_selection('backward', X_train_new, y_train, 1,model)
        logging_info('procces_full()','4/8')

        # WRAPPER: FORWARD FLOATING SELECTION
        best_features_forward_float_woc = feature_selection('forward_floating', X_train, y_train, top_feat_woc,model)
        best_features_forward_float_wc = feature_selection('forward_floating', X_train_new, y_train, top_feat_wc,model)
        logging_info('procces_full()','5/8')

        # WRAPPER: BACKWARD FLOATING SELECTION
        best_features_backward_float_woc = feature_selection('backward_floating', X_train, y_train, 1,model)
        best_features_backward_float_wc = feature_selection('backward_floating', X_train_new, y_train, 1,model)
        logging_info('procces_full()','6/8')

        # EMBEDDED: FEATURE IMPORTANCE
        best_features_importance_woc = feature_selection('feature_importance',X_train, y_train,top_feat_woc,model)
        best_features_importance_wc = feature_selection('feature_importance',X_train_new,y_train,top_feat_wc,model)
        logging_info('procces_full()','7/8')

        # HYBRID: RFE
        best_features_rfe_woc = feature_selection('RFE', X_train, y_train, top_feat_woc,model)
        best_features_rfe_wc = feature_selection('RFE', X_train_new, y_train, top_feat_wc,model)
        logging_info('procces_full()','8/8')

        features = {
            'features_without_criba': features_without_criba,
            'features_with_criba': features_with_criba,
            'features_pearson_woc': best_features_pearson_woc,
            'features_pearson_wc': best_features_pearson_wc,
            'features_mutual_woc': best_features_mutual_woc,
            'features_mutual_wc': best_features_mutual_wc,
            'features_forward_woc': best_features_forward_woc,
            'features_forward_wc': best_features_forward_wc,
            'features_backward_woc': best_features_backward_woc,
            'features_backward_wc': best_features_backward_wc,
            'features_forward_float_woc': best_features_forward_float_woc,
            'features_forward_float_wc': best_features_forward_float_wc,
            'features_backward_float_woc': best_features_backward_float_woc,
            'features_backward_float_wc': best_features_backward_float_wc,
            'features_importance_woc': best_features_importance_woc,
            'features_importance_wc': best_features_importance_wc,
            'features_rfe_woc': best_features_rfe_woc,
            'features_rfe_wc': best_features_rfe_wc
        }
        results = procces_results(features, X_train, X_test, y_train, y_test)
        self.results = results

    def procces_reduction(self):

        # CARGA DE DATOS
        url='https://raw.githubusercontent.com/Sermor95/Machine-Learning/master/datasets/'+self.dataset+'.csv'
        data = pd.read_csv(url)
        X = data.iloc[:,:-1]
        y = data.iloc[:,-1]

        # ONE HOT ENCODING
        if need_ohe(self.dataset, X):
            X = apply_one_hot_encoding(X)

        # SPLIT
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        features_without_criba = (get_top_feat_by_config(self.id,'Criba Person',False,None,False),None)
        features_with_criba = (get_top_feat_by_config(self.id,'Criba Person',True,None,False),None)

        top_feat_woc = get_top_feat(len(X.columns), self.reduction)
        top_feat_wc = get_top_feat(len(features_with_criba[0]), self.reduction)
        logging_info(f'procces_full()---top_feat_woc', top_feat_woc)
        logging_info(f'procces_full()---top_feat_wc', top_feat_wc)

        # FILTER: PEARSON CORRELATION
        best_features_pearson_woc = (get_top_feat_by_config(self.id,'Person Correlation',False,top_feat_woc,False),None)
        best_features_pearson_wc = (get_top_feat_by_config(self.id,'Person Correlation',True,top_feat_wc,False),None)
        logging_info('procces_reduction()','1/8')

        # FILTER: MUTUAL INFORMATION
        best_features_mutual_woc = (get_top_feat_by_config(self.id,'Mutual Information',False,top_feat_woc,False),None)
        best_features_mutual_wc = (get_top_feat_by_config(self.id,'Mutual Information',True,top_feat_wc,False),None)
        logging_info('procces_reduction()','2/8')

        # WRAPPER: FORWARD SELECTION
        best_features_forward_woc = (get_top_feat_by_config(self.id,'Forward Selection',False,top_feat_woc,True),None)
        best_features_forward_wc = (get_top_feat_by_config(self.id,'Forward Selection',True,top_feat_wc,True),None)
        logging_info('procces_reduction()','3/8')

        # WRAPPER: BACKWARD SELECTION
        best_features_backward_woc = (get_top_feat_by_config(self.id,'Backward Selection',False,top_feat_woc,True),None)
        best_features_backward_wc = (get_top_feat_by_config(self.id,'Backward Selection',True,top_feat_wc,True),None)
        logging_info('procces_reduction()','4/8')

        # WRAPPER: FORWARD FLOATING SELECTION
        best_features_forward_float_woc = (get_top_feat_by_config(self.id,'Forward Floating Selection',False,top_feat_woc,True),None)
        best_features_forward_float_wc = (get_top_feat_by_config(self.id,'Forward Floating Selection',True,top_feat_wc,True),None)
        logging_info('procces_reduction()','5/8')

        # WRAPPER: BACKWARD FLOATING SELECTION
        best_features_backward_float_woc = (get_top_feat_by_config(self.id,'Backward Floating Selection',False,top_feat_woc,True),None)
        best_features_backward_float_wc = (get_top_feat_by_config(self.id,'Backward Floating Selection',True,top_feat_wc,True),None)
        logging_info('procces_reduction()','6/8')

        # EMBEDDED: FEATURE IMPORTANCE
        best_features_importance_woc = (get_top_feat_by_config(self.id,'Feature Importance',False,top_feat_woc,False),None)
        best_features_importance_wc = (get_top_feat_by_config(self.id,'Feature Importance',True,top_feat_wc,False),None)
        logging_info('procces_reduction()','7/8')

        # HYBRID: RFE
        best_features_rfe_woc = (get_top_feat_by_config(self.id,'RFE',False,top_feat_woc,False),None)
        best_features_rfe_wc = (get_top_feat_by_config(self.id,'RFE',True,top_feat_wc,False),None)
        logging_info('procces_reduction()','8/8')

        features = {
            'features_without_criba': features_without_criba,
            'features_with_criba': features_with_criba,
            'features_pearson_woc': best_features_pearson_woc,
            'features_pearson_wc': best_features_pearson_wc,
            'features_mutual_woc': best_features_mutual_woc,
            'features_mutual_wc': best_features_mutual_wc,
            'features_forward_woc': best_features_forward_woc,
            'features_forward_wc': best_features_forward_wc,
            'features_backward_woc': best_features_backward_woc,
            'features_backward_wc': best_features_backward_wc,
            'features_forward_float_woc': best_features_forward_float_woc,
            'features_forward_float_wc': best_features_forward_float_wc,
            'features_backward_float_woc': best_features_backward_float_woc,
            'features_backward_float_wc': best_features_backward_float_wc,
            'features_importance_woc': best_features_importance_woc,
            'features_importance_wc': best_features_importance_wc,
            'features_rfe_woc': best_features_rfe_woc,
            'features_rfe_wc': best_features_rfe_wc
        }
        results = procces_results(features, X_train, X_test, y_train, y_test)
        self.results = results

    def getResultados(self):
        return self.results
    
    def toJSON(self):
        return json.dumps(self.__dict__)
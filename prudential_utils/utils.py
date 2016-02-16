'''
This script contains procedures for feature engineering and stacking algorithms
The basic feature enginering steps were adopted from the public script:
https://www.kaggle.com/realtwo/prudential-life-insurance-assessment/xgb-test/run/152209
On top of that, I engineered additional features:

1. Bumpers (the idea is borrowed from the 2d place solution of CrowdFowler Search Relevance: https://github.com/geffy/kaggle-crowdflower)
   In short Bumpers are binary probabilities estimated on different splits of labels (i.e., 1 vs rest, 1 and 2 vs rest, 1, 2, and 3 vs rest, etc.)
   These binary classifiers were trained using 4-fold CV. Overall, there are 7 bumpers for 8-label data.

2. Stacked probabilities of multi-label classifier (XGBoost) obtained using 4-fold stacked generalization.

'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from ml_metrics import quadratic_weighted_kappa
from sklearn.cross_validation import StratifiedKFold

from prudential_utils import paths

path_train = paths.DATA_TRAIN_PATH
path_test=paths.DATA_TEST_PATH
path_sample_submission=paths.SAMPLE_SUBMISSION_PATH
path_submission=paths.SUBMISSION_PATH

def Load_data():
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    # combine train and test
    data_comb = train.append(test)

    # Found at https://www.kaggle.com/marcellonegro/prudential-life-insurance-assessment/xgb-offset0501/run/137585/code
    # create any new variables    
    data_comb['Product_Info_2_char'] = data_comb.Product_Info_2.str[0]
    data_comb['Product_Info_2_num'] = data_comb.Product_Info_2.str[1]

    # factorize categorical variables
    data_comb['Product_Info_2'] = pd.factorize(data_comb['Product_Info_2'])[0]
    data_comb['Product_Info_2_char'] = pd.factorize(data_comb['Product_Info_2_char'])[0]
    data_comb['Product_Info_2_num'] = pd.factorize(data_comb['Product_Info_2_num'])[0]

    data_comb['BMI_Age'] = data_comb['BMI'] * data_comb['Ins_Age']

    med_keyword_columns = data_comb.columns[data_comb.columns.str.startswith('Medical_Keyword_')]
    data_comb['Med_Keywords_Count'] = data_comb[med_keyword_columns].sum(axis=1)

    print('Encode missing values')    
    data_comb.fillna(-1, inplace=True)

    # fix the dtype on the label column
    data_comb['Response'] = data_comb['Response'].astype(int)

    # split train and test
    train = data_comb[data_comb['Response']>0].copy()
    test = data_comb[data_comb['Response']<1].copy()

    target = train['Response'].values 
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(target) 

    train.drop(['Id', 'Response', 'Medical_History_10','Medical_History_24'], axis=1, inplace=True)
    test.drop(['Id', 'Response', 'Medical_History_10','Medical_History_24'], axis=1, inplace=True)
    train = train.as_matrix()
    test = test.as_matrix()

    print('Construct labels for bumping')
    num_class = len(np.unique(target))
    labels = np.zeros(shape=(train.shape[0],num_class-1))
    labels[:, 0][target==1]=1
    labels[:, 6][target<8]=1
    for i in range(1, num_class-2):
        labels[:, i][target<i+2]=1
    return train, test, target, labels   

def save_submission(predictions):
    sample_submission = pd.read_csv(path_sample_submission)
    sample_submission['Response'] = predictions
    sample_submission.to_csv(path_submission, index=False) 

def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return quadratic_weighted_kappa(yhat, y)

def Stack_Bump_Probs(train, test, clfs, labels, n_folds): # train data, test data, list of classifiers,
                                                # matrix of bumping labels, number of folders

    print("Generating Meta-features")
    blend_train = np.zeros((train.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((test.shape[0], len(clfs)))   # Number of testing data x Number of classifiers
    
    for j, clf in enumerate(clfs):
        
        print ('Training classifier [%s]' % (j))
        skf = list(StratifiedKFold(labels[:, j], n_folds))
        for i, (tr_index, cv_index) in enumerate(skf):
            
            print ('stacking Fold [%s] of train data' % (i))
            
            # This is the training and validation set (train on 2 folders, predict on a 3d folder)
            X_train = train[tr_index]
            Y_train = labels[:, j][tr_index]
            X_cv = train[cv_index]
            clf.fit(X_train, Y_train)                                
            pred = clf.predict_proba(X_cv)
            blend_train[cv_index, j] = pred
                 
        print('stacking test data') 
        clf.fit(train, labels[:, j])
        pred = clf.predict_proba(test)
        
        blend_test[:, j] = pred
           
    return blend_train, blend_test      
      
def Stack_Multi(train, test, y, clfs, n_folds, scaler=None): # train data, test data, Target data,
                                                # list of models to stack, number of folders, boolean for scaling

    print("Generating Meta-features")
    num_class = np.unique(y).shape[0]
    skf = list(StratifiedKFold(y, n_folds))
    if scaler:
        scaler = StandardScaler().fit(train)
        train_sc = scaler.transform(train)
        test_sc = scaler.transform(test)
    else:
        train_sc = train
        test_sc = test
    blend_train = np.zeros((train.shape[0], num_class*len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((test.shape[0], num_class*len(clfs)))   # Number of testing data x Number of classifiers   
    for j, clf in enumerate(clfs):
        print ('Training classifier [%s]' % (j))
        for i, (tr_index, cv_index) in enumerate(skf):
            
            print ('stacking Fold [%s] of train data' % (i))
            
            # This is the training and validation set (train on 2 folders, predict on a 3d folder)
            X_train = train[tr_index]
            Y_train = y[tr_index]
            X_cv = train[cv_index]
            if scaler:
               scaler_cv = StandardScaler().fit(X_train)
               X_train=scaler_cv.transform(X_train)
               X_cv=scaler_cv.transform(X_cv)
            clf.fit(X_train, Y_train)
            pred = clf.predict_proba(X_cv)
            blend_train[cv_index, j*num_class:(j+1)*num_class] = pred
                
        print('stacking test data')        
        clf.fit(train_sc, y)
        pred = clf.predict_proba(test_sc)

        blend_test[:, j*num_class:(j+1)*num_class] = pred
                   
    return blend_train, blend_test
    
def Stack_Regr(train, test, y, clfs, n_folds): # train data, test data, Target data,
                                                # list of models to stack, number of folders

    print("Generating Meta-features")
    skf = list(StratifiedKFold(y, n_folds))
    blend_train = np.zeros((train.shape[0], len(clfs))) # Number of training data x Number of classifiers
    blend_test = np.zeros((test.shape[0], len(clfs)))   # Number of testing data x Number of classifiers   
    for j, clf in enumerate(clfs):
        print ('Training classifier [%s]' % (j))
        for i, (tr_index, cv_index) in enumerate(skf):
            
            print ('stacking Fold [%s] of train data' % (i))
            
            # This is the training and validation set (train on 2 folders, predict on a 3d folder)
            X_train = train[tr_index]
            Y_train = y[tr_index]
            X_cv = train[cv_index]
            clf.fit(X_train, Y_train)
            pred = clf.predict(X_cv)
            blend_train[cv_index, j] = pred
                
        print('stacking test data')        
        clf.fit(train, y)
        pred = clf.predict(test)

        blend_test[:, j] = pred
                   
    return blend_train, blend_test    
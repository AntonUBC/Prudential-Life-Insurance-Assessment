'''
This script produces custom wrappers for XGBoost models
Description:

1. XGBoost_multilabel: gradient boosting model for estimation of multi-class probabilities.

2. XGBoost_binary: gradient boosting model for estimation of binary probabilities. 

3. XGBoost_regressor: gradient boosting model for regression. 
'''

import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.base import BaseEstimator


class XGBoost_multilabel(BaseEstimator):
    def __init__(self, nthread, eta,
                 gamma, max_depth, min_child_weight, max_delta_step,
                 subsample, colsample_bytree, silent, seed,
                 l2_reg, l1_reg, num_round):
        self.silent = silent
        self.nthread = nthread
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.silent = silent
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.num_round=num_round
        self.num_class = None
        self.model = None

    def fit(self, X, y):
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)
        self.num_classes = np.unique(y).shape[0]
        sf = xgb.DMatrix(X, y)
        params = {"objective": 'multi:softprob',
          "eta": self.eta,
          "gamma": self.gamma,
          "max_depth": self.max_depth,
          "min_child_weight": self.min_child_weight,
          "max_delta_step": self.max_delta_step,
          "subsample": self.subsample,
          "silent": self.silent,
          "colsample_bytree": self.colsample_bytree,
          "seed": self.seed,
          "lambda": self.l2_reg,
          "alpha": self.l1_reg,
          "num_class": self.num_classes}
        self.model = xgb.train(params, sf, self.num_round)

        return self

    def predict_proba(self, X):
        X=xgb.DMatrix(X)
        preds = self.model.predict(X)
        return preds

class XGBoost_binary(BaseEstimator):
    def __init__(self, nthread, eta,
                 gamma, max_depth, min_child_weight, max_delta_step,
                 subsample, colsample_bytree, scale_pos_weight, silent, seed,
                 l2_reg, l1_reg, n_estimators):
        self.silent = silent
        self.nthread = nthread
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.silent = silent
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.seed = seed
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.n_estimators=n_estimators
        self.model = None

    def fit(self, X, y):
        sf = xgb.DMatrix(X, y)
        params = {"objective": 'binary:logistic',
                  "eval_metric": 'auc',
          "eta": self.eta,
          "gamma": self.gamma,
          "max_depth": self.max_depth,
          "min_child_weight": self.min_child_weight,
          "max_delta_step": self.max_delta_step,
          "subsample": self.subsample,
          "silent": self.silent,
          "colsample_bytree": self.colsample_bytree,
          "scale_pos_weight": self.scale_pos_weight,
          "seed": self.seed,
          "lambda": self.l2_reg,
          "alpha": self.l1_reg}
        self.model = xgb.train(params, sf, self.n_estimators)
        return self

    def predict_proba(self, X):
        X=xgb.DMatrix(X)
        preds = self.model.predict(X)
        return preds
        
class XGBoost_regressor(BaseEstimator):
    def __init__(self, nthread, eta,
                 gamma, max_depth, min_child_weight, max_delta_step,
                 subsample, colsample_bytree, silent, seed,
                 l2_reg, l1_reg, n_estimators):
        self.silent = silent
        self.nthread = nthread
        self.eta = eta
        self.gamma = gamma
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.silent = silent
        self.colsample_bytree = colsample_bytree
        self.seed = seed
        self.l2_reg = l2_reg
        self.l1_reg = l1_reg
        self.n_estimators=n_estimators
        self.model = None

    def fit(self, X, y):
        sf = xgb.DMatrix(X, y)
        params = {"objective": 'reg:linear',
                  "eta": self.eta,
          "gamma": self.gamma,
          "max_depth": self.max_depth,
          "min_child_weight": self.min_child_weight,
          "max_delta_step": self.max_delta_step,
          "subsample": self.subsample,
          "silent": self.silent,
          "colsample_bytree": self.colsample_bytree,
          "seed": self.seed,
          "lambda": self.l2_reg,
          "alpha": self.l1_reg}
        self.model = xgb.train(params, sf, self.n_estimators)
        return self

    def predict(self, X):
        X=xgb.DMatrix(X)
        preds = self.model.predict(X)
        return preds        
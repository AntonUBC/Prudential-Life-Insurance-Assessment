'''
This script generates meta-probabilities and bumping probabilities, trains three separate models, ensembles their predictions,
and train offset values to construct the vector of predicted classes for the final submission.

The offset learning idea was borrowed (with substantial changes) from a public script by Michael Hartmann:
https://www.kaggle.com/zeroblue/prudential-life-insurance-assessment/xgboost-with-optimized-offsets/run/133836
However, as opposed to that script, offsets were trained on STACKED training predictions, which reduces the chance
of overfitting. Moreover, the initial values of offsets were chosen based on the discrepancies between test predictions
and the distribution of labels in training data estimated at quantile values.

'''

import numpy as np
from scipy.optimize import fmin_powell
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

# load user modules
from prudential_utils import utils
from wrappers import models

n_folds = 4 # set the number of folders for generating meta-features

def apply_offset(data, bin_offset, sv, scorer=utils.eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score

print('Load data')
train, test, target, labels = utils.Load_data()

clf1 = models.XGBoost_binary(nthread=6, eta=0.003 ,gamma=1.2, max_depth=6,
                         min_child_weight=2, max_delta_step=None,
                         subsample=0.6, colsample_bytree=0.35, scale_pos_weight=1.5,
                         silent=0, seed=1301, l2_reg=1, l1_reg=0.2, n_estimators=4269)
                         
clf2 = models.XGBoost_binary(nthread=6, eta=0.004 ,gamma=1.2, max_depth=6,
                         min_child_weight=2, max_delta_step=None,
                         subsample=0.6, colsample_bytree=0.35, scale_pos_weight=1,
                         silent=0, seed=1301, l2_reg=1, l1_reg=0.2, n_estimators=4200)
                         
clf3 = models.XGBoost_binary(nthread=6, eta=0.004 ,gamma=1.2, max_depth=6,
                         min_child_weight=2, max_delta_step=None,
                         subsample=0.6, colsample_bytree=0.35, scale_pos_weight=1,
                         silent=0, seed=1301, l2_reg=1, l1_reg=0.2, n_estimators=4190)

clf4 = models.XGBoost_binary(nthread=6, eta=0.004 ,gamma=1.2, max_depth=6,
                         min_child_weight=2, max_delta_step=None,
                         subsample=0.6, colsample_bytree=0.35, scale_pos_weight=1,
                         silent=0, seed=1301, l2_reg=1, l1_reg=0.2, n_estimators=4188)

clf5 = models.XGBoost_binary(nthread=6, eta=0.004 ,gamma=1.2, max_depth=6,
                         min_child_weight=2, max_delta_step=None,
                         subsample=0.6, colsample_bytree=0.35, scale_pos_weight=1,
                         silent=0, seed=1301, l2_reg=1, l1_reg=0.2, n_estimators=4191)

clf6 = models.XGBoost_binary(nthread=6, eta=0.004 ,gamma=0.95, max_depth=6,
                         min_child_weight=4, max_delta_step=None,
                         subsample=0.55, colsample_bytree=0.35, scale_pos_weight=1,
                         silent=0, seed=1301, l2_reg=1, l1_reg=0.3, n_estimators=4190)

clf7 = models.XGBoost_binary(nthread=6, eta=0.004 ,gamma=0.85, max_depth=7,
                         min_child_weight=4, max_delta_step=None,
                         subsample=0.6, colsample_bytree=0.3, scale_pos_weight=1,
                         silent=0, seed=1301, l2_reg=1, l1_reg=0.05, n_estimators=4290)

clfs1 = [clf1, clf2, clf3, clf4, clf5, clf6, clf7]                         

print('Compute bumping probabilities')
bumps_train, bumps_test = utils.Stack_Bump_Probs(train, test, clfs1, labels, n_folds)

clf8 =  models.XGBoost_multilabel(nthread=6, eta=0.012,
                 gamma=1, max_depth=6, min_child_weight=10, max_delta_step=0,
                 subsample=0.65, colsample_bytree=0.5, silent=1, seed=1301,
                 l2_reg=1.5, l1_reg=0, num_round=975)                                      

clfs2 = [clf8]
print('Compute stacking probabilities')
y = (target-1)
train_probs_xgb, test_probs_xgb = utils.Stack_Multi(np.column_stack((train, bumps_train)), np.column_stack((test, bumps_test)), y, clfs2, n_folds) 
print('Construct stacking data')
train_stuck = np.column_stack((train, bumps_train, train_probs_xgb))
test_stuck = np.column_stack((test, bumps_test, test_probs_xgb))

clf9 = models.XGBoost_regressor(nthread=3, eta=0.0057, gamma=0, max_depth=6,
                                min_child_weight=2, max_delta_step=None,
                                subsample=0.66, colsample_bytree=0.7,
                                silent=1, seed=1301, l2_reg=0, l1_reg=0, n_estimators=1000)
                 
clf10 = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=6, min_samples_split=2,
                               min_samples_leaf=4, min_weight_fraction_leaf=0.0,
                               max_features=0.5, max_leaf_nodes=None, bootstrap=True, 
                               oob_score=False, n_jobs=3, random_state=1301)
                               
clf11 = ExtraTreesRegressor(n_estimators=1000, criterion='mse', max_depth=6, min_samples_split=2,
                               min_samples_leaf=4, min_weight_fraction_leaf=0.0,
                               max_features=0.62, max_leaf_nodes=None, bootstrap=False,  
                               oob_score=False, n_jobs=3, random_state=1301)
                               
clfs3 = [clf9, clf10, clf11]  

print('Train ensemble models')
train_preds_stuck, test_preds_stuck = utils.Stack_Regr(train_stuck, test_stuck, target, clfs3, n_folds)
print('Compute ensemble predictions')

# Stacked train data is used for training offset values, this reduces the chance of overfitting
train_preds = (train_preds_stuck[:,0]**0.41)*(train_preds_stuck[:,1]**0.01)*(train_preds_stuck[:,2]**0.58)
test_preds = (test_preds_stuck[:,0]**0.41)*(test_preds_stuck[:,1]**0.01)*(test_preds_stuck[:,2]**0.58)

print('Train offset values for label construction')
num_classes=np.unique(target).shape[0]

# Compute quantiles of test predictions
quant = []
for q in range(1, 100):
    p = np.percentile(test_preds, q)
    quant.append(p)

#  Compute initial offset values based on the discrepancies between label distribution of train data and the distribution of test predictions    
offsets = -1*np.array([quant[9] - 1.5 ,quant[20] - 2.5,quant[22] - 3.5,quant[24] - 4.5,0,quant[34] - 5.5,quant[53] - 6.5,quant[67] - 7.5]) 

# train offsets 
data = np.vstack((train_preds, train_preds, target))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
for j in range(num_classes):
    train_offset = lambda x: -apply_offset(data, x, j)
    offsets[j] = fmin_powell(train_offset, offsets[j])  

print('Apply offsets to test')
data = np.vstack((test_preds, test_preds))
for j in range(num_classes):
    data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

preds_subm = np.round(np.clip(data[1], 1, 8)).astype(int)

# Save submission
print('Save submission file')  
utils.save_submission(preds_subm) 
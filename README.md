## Solution to [Prudential Life Insurance Assessment Challenge] (https://www.kaggle.com/c/prudential-life-insurance-assessment)

This is the model which I unfortunately did not select for submission. This model gives the quadratic weighted kappa score of 0.67456 and 65th place on the leaderboard (top 3%).
Ironically, this is more than 700 positions higher than the rank of my official submission which badly overfitted to the LB data.

### Project Description

The task was to develop a predictive model that accurately classifies the insurance risk and help Prudential better understand 
the predictive power of the data points in the existing assessment, enabling them to significantly streamline the insurance process.

### Data

Over a hundred variables describing attributes of life insurance applicants. The task was to predict the "Response" variable for each
Id in the test set. "Response" is an ordinal measure of risk that has 8 levels.

### Data preprocessing and feature engineering

Data preprocessing was fairly standard: label encoding, replacing NAs with -1, and etc. I also generated some additional features
to increase the predictive power of the model:
  - Bumpers - predicted probabilities of a binary classifier (XGBoost) for different label splits (e.g., 1 vs rest, 1 and 2 vs rest, 1, 2, and 3 vs rest, and etc. Overall, 7 bumpers).
  This idea was borrowed from the [2d place solution of the Kaggle CrowdFlower challenge] (https://github.com/geffy/kaggle-crowdflower). By construction, these bumpers should reduce the predictive error of the model.

  - Stucked predictions of a multi-label classifier (XGBoost) constructed using a 4-fold stacked generalization. 

### Solution

 The model is an ensemble of three regression models: Gradient Boosting Trees (XGBoost), Random Forest (sklearn), and Extremely Randomized Trees (sklearn).
 Both hyperparameters of ensembled models and ensemble weights were trained using 4-fold cross-validation.
 The idea for mapping regressor predictors to labels was borrowed from the [public script] (https://www.kaggle.com/zeroblue/prudential-life-insurance-assessment/xgboost-with-optimized-offsets/run/133836) of Michael Hartman.
 The purpose of this procedure is to use training data to estimate the label-specific shifters (offsets) which should be applied to predicted values
 in order to reduce the prediction error. However, I made substantial changes to the procedure:
   - I trained the offset values using the stacked train predictions instead of fitted values of the model used in the original approach. This should have reduce the chance of overfitting (and it did!).
   
   - I chose the initial offset values based on the discrepancies between test predictions and the distribution of labels in training data estimated at quantile values.
   
### Instruction

Download the project folder to your computer and run the file ```/prudential_risk_prediction/ensemble/ensemble_submission.py```
(you may need to adjust paths in ```prudential_risk_prediction/prudential_utils/paths.py``` accordingly). This will generate
the submission file in csv format and load it to ```prudential_risk_prediction/data/submission``` folder. This file can be then submitted at
[the project webpage] (https://www.kaggle.com/c/prudential-life-insurance-assessment). Alternatively, you can submit an existing file.

Scripts:
- ```/prudential_risk_prediction/wrappers/models.py``` contains wrapper-classes for XGBoost (to make it more sklearn-like). This module is used by ```ensemble_submission.py```.
- ```/prudential_risk_prediction/prudential_utils/utils.py``` contains functions which are used for data load, feature engineering, stacking, and saving the submission file in csv format. This module is used by ```ensemble_submission.py```.
- ```/prudential_risk_prediction/prudential_utils/paths.py``` contains paths to data and submission folders.
- ```/prudential_risk_prediction/ensemble/ensemble_submission.py``` is the main module which is used to generate predictions. 

### Dependencies
- Python 3.4 (Python 2.7 would also work, just type: ```from __future__ import print_function``` in the beginning of the script)

- Pandas (any relatively recent version would work)

- Numpy (any relatively recent version would work)

- Sklearn (any relatively recent version would work)

- XGBoost 0.4.0

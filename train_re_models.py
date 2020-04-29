"""
This is the .py version of train_models_win.ipynb.
It could runs in the raspbeery pi

1. Use 20 features and 4 labels from features.csv
2. Try different models
"""
# ---------------------------------------------
# 1. Initialization
# ---------------------------------------------
from enum import Enum
features_path = r"./dataset/features_2.csv"
model_folder = r"./models/large_models_2"
class FingerType(Enum):
    Thumb = 0
    Index = 1

finger_type = FingerType.Thumb

print("Import Packages...")

import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
import joblib 
import pickle
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, BaggingRegressor, \
BaggingRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, \
VotingRegressor
# import lightgbm as lgb
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from copy import deepcopy

# ---------------------------------------------
# 2. Prepare data
# ---------------------------------------------
print("\nReading Data...")

features_pd = pd.read_csv(features_path)
X = features_pd.iloc[:, 0: 20].to_numpy()
if finger_type == FingerType.Thumb:
    Y =  features_pd.iloc[:, 20: 22].to_numpy()
elif finger_type == FingerType.Index:
    Y =  features_pd.iloc[:, 22: 24].to_numpy()

print("Original dataset shape")
print("X:", X.shape, " Y:", Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
print("Train dataset shape")
print("X:", X_train.shape, " Y:", Y_train.shape)
print("Test dataset shape")
print("X:", X_test.shape, " Y:", Y_test.shape)

# ---------------------------------------------
# 3. Prepare the regression model training pipeline
# ---------------------------------------------
def generate_joint_model(single_model):
    model = MultiOutputRegressor(single_model)
    model.fit(X_train, Y_train)
    
    score_train = model.score(X_train, Y_train)
    print('Score of train', round(score_train * 100, 1), "%")
    
    score = model.score(X_test, Y_test)
    print('Score of test', round(score * 100, 1), "%")
    
    model_path = model_folder + r"/" +  \
                    str(round(score, 3)).replace('.', '_') + r"_" +  \
                    str(model.get_params()['estimator']).split('(')[0] + \
                    '.joblib'
    joblib.dump(model, model_path)
    print("Save model file", model_path)
    
    return model, model_path

def generate_single_models(single_model):
    models = []
    file_label = ['x', 'y']
    scores = []
    file_names = []
    for i in range(2):
        model = deepcopy(single_model)    
        model.fit(X_train, Y_train[:, i])
    
        models.append(model)
        
        score_train = model.score(X_train, Y_train[:, i])
        score_test  = model.score(X_test, Y_test[:, i])
        
        single_score = {'train': score_train, 'test': score_test}
        scores.append(single_score)
        
        model_path = model_folder + r'/' +  \
                        str(round(score_test, 3)).replace('.', '_') + r'_' +  \
                        str(type(single_model)).split('.')[-1].split('\'')[0] + r'_' + \
                        file_label[i] + '.joblib'
        joblib.dump(model, model_path)
        file_names.append(model_path)
        
    for i in range(2):
        print(file_label[i])
        print('Score of train', round(scores[i]['train'] * 100, 1), "%")
        print('Score of test',  round(scores[i]['test']  * 100, 1), "%")
        print("Save model file", file_names[i])
    
    return models, file_names

# ---------------------------------------------
# 4. Use specific model
# ---------------------------------------------
# single_model = SVR(kernel='linear', C=1.0, epsilon=0.2, max_iter=10000)
# single_model = SVR(kernel='poly', C=1.0, epsilon=0.2, max_iter=10000)
# single_model = SVR(kernel='sigmoid', C=1.0, epsilon=0.2, tol=0.1)
# single_model = LogisticRegression(random_state=0)

# single_model = RandomForestRegressor(n_estimators=60, max_depth=20, 
#                                      min_samples_split=15, min_samples_leaf=15,
#                                      verbose=3)
single_model = RandomForestRegressor(n_estimators=100, max_depth=20,min_samples_leaf=15, verbose=0)                                     
# single_model = RandomForestRegressor(n_estimators=60, max_depth=None, criterion='mae', min_samples_split=15, min_samples_leaf=15)

# single_model = LinearRegression()

# single_model = AdaBoostRegressor(random_state=0, n_estimators=100, loss='square')
# single_model = BaggingRegressor(base_estimator=SVR(), n_estimators=10, random_state=0)

# ** single_model = GradientBoostingRegressor(n_estimators=100)
# ** single_model = HistGradientBoostingRegressor(max_iter=10000)

# r1 = RandomForestRegressor(max_depth=None, random_state=None)
# r2 = GradientBoostingRegressor(n_estimators=10000)
# r3 = HistGradientBoostingRegressor(max_iter=10000)
# ** single_model = VotingRegressor([('RF', r1), ('GB', r2), ('HGB', r3)])

# LR1 = LinearRegression()
# LR2 = LinearRegression()
# LR3 = LinearRegression()
# single_model = VotingRegressor([('LR1', r1), ('LR2', r2), ('LR3', r3)])

# ** single_model = lgb.LGBMRegressor(boosting_type='dart', num_leaves=100, n_estimators=10000)

# single_model = MLPRegressor()

# kernel = DotProduct() + WhiteKernel()
# single_model = GaussianProcessRegressor(kernel=kernel, random_state=0)

# ---------------------------------------------
# 5. Start Training
# ---------------------------------------------
print("\nStart Training...")
# model, model_path = generate_model(single_model)
models, model_paths = generate_single_models(single_model)

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
from HyperparameterTuning import getParams
from Testing import getTestResults
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
BEST_F1 = 0
BEST_VAR = 0
MIN_SAMPLES_SPLIT = 9
MAX_DEPTH_TREE = 20
  
X = pd.read_csv('ann-train.data', header=None, sep = ' ').drop([22,23],axis=1)
y = X[21]
X = X.drop(21,axis=1)
print(X)
X_train = X.to_numpy()
y_train = y.to_numpy().flatten()

X = pd.read_csv('ann-test.data', header=None, sep = ' ').drop([22,23],axis=1)
y = X[21]
X = X.drop(21,axis=1)
print(X)
X_test = X.to_numpy()
y_test = y.to_numpy().flatten()
print(f"Number of samples: {X_train.shape}")
print(f"Train data distribution: {np.unique_counts(y_train)}")
print(f"Test data distribution: {np.unique_counts(y_test)}")
#obtaining tuned hyperparameters
MIN_SAMPLES_SPLIT, MAX_DEPTH_TREE, BEST_F1, BEST_VAR = getParams(X_train, y_train)
print(f"Tuned hyperparameters: ")
print(f"Minimum number of samples per node: {MIN_SAMPLES_SPLIT}")
print(f"Maximum Depth: {MAX_DEPTH_TREE}")
print(f"F1 Score on train data: {BEST_F1}, and variance: {BEST_VAR}")
#Saving
PATH = os.getcwd()
RESULT_PATH_TRAIN = PATH + '/CAUSALMETRIC-TUNING/Thyroid/'


try:
    os.makedirs(RESULT_PATH_TRAIN)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH_TRAIN)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_TRAIN)

np.save(RESULT_PATH_TRAIN+"/h_min_samples_split.npy", np.array([MIN_SAMPLES_SPLIT]) ) 
np.save(RESULT_PATH_TRAIN+"/h_max_depth.npy", np.array([MAX_DEPTH_TREE]) ) 
np.save(RESULT_PATH_TRAIN+"/h_variance.npy", np.array([BEST_VAR]) ) 
np.save(RESULT_PATH_TRAIN+"/h_F1SCORE.npy", np.array([BEST_F1]) ) 


ACCURACY = 0
F1_SCORE = 0
PRECISION = 0
RECALL = 0

RESULT_PATH_TEST = PATH + '/CAUSALMETRIC-TESTING/Thyroid/'

ACCURACY,F1_SCORE, PRECISION, RECALL = getTestResults(X_train, y_train, MIN_SAMPLES_SPLIT, MAX_DEPTH_TREE,X_test, y_test,'Thyroid')
try:
    os.makedirs(RESULT_PATH_TEST)
except OSError:
    print ("Creation of the result directory %s not required" % RESULT_PATH_TEST)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_TEST)

np.save(RESULT_PATH_TEST+"/h_ACCURACY.npy", np.array([ACCURACY]) ) 
np.save(RESULT_PATH_TEST+"/h_F1 SCORE.npy", np.array([F1_SCORE]) ) 
np.save(RESULT_PATH_TEST+"/h_PRECISION.npy", np.array([PRECISION]) ) 
np.save(RESULT_PATH_TEST+"/h_RECALL.npy", np.array([RECALL])) 
print("Accuarcy is: ", ACCURACY)
print("F1 Score is: ", F1_SCORE)
print("Precision is: ", PRECISION)
print("Recall is: ", RECALL)
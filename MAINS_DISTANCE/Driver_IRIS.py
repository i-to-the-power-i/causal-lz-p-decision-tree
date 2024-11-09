
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
from HyperparameterTuning import getParams
from Testing import getTestResults
BEST_F1 = 0
BEST_VAR = 0
MIN_SAMPLES_SPLIT = 0
MAX_DEPTH_TREE = 0

iris = load_iris()
#reading data and labels from the dataset
X = iris.data
y = iris.target

#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

#obtaining tuned hyperparameters
MIN_SAMPLES_SPLIT, MAX_DEPTH_TREE, BEST_F1, BEST_VAR = getParams(X_train, y_train)
print(f"Tuned hyperparameters: ")
print(f"Minimum number of samples per node: {MIN_SAMPLES_SPLIT}")
print(f"Maximum Depth: {MAX_DEPTH_TREE}")
print(f"F1 Score on train data: {BEST_F1}, and variance: {BEST_VAR}")
#Saving
PATH = os.getcwd()
RESULT_PATH_TRAIN = PATH + '/DISTANCEMETRIC-TUNING/IRIS/'


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

RESULT_PATH_TEST = PATH + '/DISTANCEMETRIC-TESTING/IRIS/'

ACCURACY,F1_SCORE, PRECISION, RECALL = getTestResults(X_train, y_train, MIN_SAMPLES_SPLIT, MAX_DEPTH_TREE,X_test, y_test,'iris')
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





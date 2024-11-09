
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os
from HyperparameterTuning import getParams
from Testing import getTestResults
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
BEST_F1 = 0
BEST_VAR = 0
MIN_SAMPLES_SPLIT = 2
MAX_DEPTH_TREE = 12
  
X = pd.read_csv('CarEvalData.csv')
y = pd.read_csv('CarEvalTargets.csv')
column_labels_list = X.columns.tolist()
print(column_labels_list)


X = X.to_numpy()
y = y.to_numpy().flatten()

# X_majority = X[y == 2]
# X_minority = X[y != 2]

# y_majority = y[y == 2]
# y_minority = y[y != 2]

# # Downsample the majority class to 200 samples
# X_majority_downsampled, y_majority_downsampled = resample(X_majority, y_majority,
#                                                          replace=False,  # no replacement
#                                                          n_samples=65,  # downsample to 200 samples
#                                                          random_state=42)

# # Combine the downsampled majority class with the minority class
# X_balanced = np.concatenate([X_majority_downsampled, X_minority], axis=0)
# y_balanced = np.concatenate([y_majority_downsampled, y_minority], axis=0)

# # Shuffle the combined data to mix the rows
# indices = np.arange(X_balanced.shape[0])
# np.random.seed(45)
# np.random.shuffle(indices)

# X_balanced = X_balanced[indices]
# y_balanced = y_balanced[indices]


# X_majority = X_balanced[y_balanced == 0]
# X_minority = X_balanced[y_balanced != 0]

# y_majority = y_balanced[y_balanced == 0]
# y_minority = y_balanced[y_balanced != 0]

# # Downsample the majority class to 200 samples
# X_majority_downsampled, y_majority_downsampled = resample(X_majority, y_majority,
#                                                          replace=False,  # no replacement
#                                                          n_samples=65,  # downsample to 200 samples
#                                                          random_state=42)

# # Combine the downsampled majority class with the minority class
# X_balanced = np.concatenate([X_majority_downsampled, X_minority], axis=0)
# y_balanced = np.concatenate([y_majority_downsampled, y_minority], axis=0)

# # Shuffle the combined data to mix the rows
# indices = np.arange(X_balanced.shape[0])
# np.random.seed(98)
# np.random.shuffle(indices)

# X_balanced = X_balanced[indices]
# y_balanced = y_balanced[indices]
# print(np.unique_counts(y_balanced))
#Splitting the dataset for training and testing (80-20)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)
print(X)
print(y[y==2].shape)
#obtaining tuned hyperparameters
MIN_SAMPLES_SPLIT, MAX_DEPTH_TREE, BEST_F1, BEST_VAR = getParams(X_train, y_train)
print(f"Tuned hyperparameters: ")
print(f"Minimum number of samples per node: {MIN_SAMPLES_SPLIT}")
print(f"Maximum Depth: {MAX_DEPTH_TREE}")
print(f"F1 Score on train data: {BEST_F1}, and variance: {BEST_VAR}")
#Saving
PATH = os.getcwd()
RESULT_PATH_TRAIN = PATH + '/DISTANCEMETRIC-TUNING/CarEval/'


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

RESULT_PATH_TEST = PATH + '/DISTANCEMETRIC-TESTING/CarEval/'

ACCURACY,F1_SCORE, PRECISION, RECALL = getTestResults(X_train, y_train, MIN_SAMPLES_SPLIT, MAX_DEPTH_TREE,X_test, y_test,'CarEval',column_labels_list)
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

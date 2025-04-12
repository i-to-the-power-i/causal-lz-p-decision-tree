#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16, 2024

"""


import os
import numpy as np




# import scipy
from scipy import io




# from sklearn.model_selection import KFold
# from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt

from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score)
from LZ_causal_measure import (calc_penalty)





PATH = os.getcwd()

#### Class - 0 represents independent process
#### Class  - 1 represents depepndent process
DATA_NAME = 'AR-20'
LAG = 20
no_of_bins=2 # No. of bins
TRIALS = 1000 ## MAXIMUM 1000 trials
VAR = 1000 # MAXIMUM no of data instances is 1000
LEN_VAL = 2000 # MAXIUM Length of the timeseries is 2000, transients are already removed

COUP_COEFF1 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
Y_causes_X = np.zeros(len(COUP_COEFF1))
X_causes_Y = np.zeros(len(COUP_COEFF1))
Y_causes_X_var = np.zeros(len(COUP_COEFF1))
X_causes_Y_var = np.zeros(len(COUP_COEFF1))

ROW = -1
for COUP_COEFF in COUP_COEFF1:
    
    ROW = ROW+1

    RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' +str(LAG)+'/' + str(COUP_COEFF) +'/'

    #### Class - 0 represents independent process
    #### Class  - 1 represents depepndent process

    Y_independent_data = io.loadmat(RESULT_PATH + 'Y_independent_data_class_0.mat')
    Y_independent_label = io.loadmat(RESULT_PATH + 'Y_independent_label_class_0.mat')
    
    X_dependent_data = io.loadmat(RESULT_PATH + 'X_dependent_data_class_1.mat' )
    X_dependent_label = io.loadmat(RESULT_PATH + 'X_dependent_label_class_1.mat')
    
    class_0_data = Y_independent_data['class_0_indep_raw_data'][0:VAR, 0:LEN_VAL]
    class_0_label = Y_independent_label['class_0_indep_raw_data_label'][0:VAR, 0:LEN_VAL]
    class_1_data = X_dependent_data['class_1_dep_raw_data'][0:VAR, 0:LEN_VAL]
    class_1_label = X_dependent_label['class_1_dep_raw_data_label'][0:VAR, 0:LEN_VAL]

    # total_data = np.concatenate((class_0_data, class_1_data))
    # total_label = np.concatenate((class_0_label, class_1_label))
    
    
    
    print("Coupling-Coefficient", COUP_COEFF)
 
    
    Y_X =[]
    X_Y = []
    for data_instance_no in range(0, TRIALS):
        
        string_X = ""
        string_Y = ""
        X = class_1_data[data_instance_no,:] # dependent data
        Y = class_0_data[data_instance_no,:] # independent data

        #generating binary sequence
        max_x = np.max(X)
        max_y = np.max(Y)
        min_x = np.min(X)
        min_y = np.min(Y)
        
        edges_x = []
        edges_x.append(min_x)

        for i in range(no_of_bins - 1):
          edge = min_x + ((i+1)*(max_x-min_x)/no_of_bins)
          edges_x.append(edge)
        # print(edges_x, max_x,min_x)

        for t in range(len(X)):
          for edge in range(len(edges_x)-1):
            if X[t] >= edges_x[edge] and X[t] <= edges_x[edge+1]:
              string_X += f"{edge}"

          if X[t] > edges_x[no_of_bins-1]:
            string_X += f"{no_of_bins-1}"



        edges_y = []
        edges_y.append(min_y)
        for i in range(no_of_bins - 1):
          edge = min_y + ((i+1)*(max_y-min_y)/no_of_bins)
          edges_y.append(edge)
        # print(bin_x, min_x)
        for t in range(len(Y)):
          for edge in range(len(edges_y)-1):
            if Y[t] >= edges_y[edge] and Y[t]<= edges_y[edge+1]:
              string_Y += f"{edge}"

          if Y[t] > edges_y[no_of_bins-1]:
            string_Y += f"{no_of_bins-1}"
        # print(string_X)


        #calculating from penalty x to y and from y to x along w strength
        # strength = 0
        # direction = 0

        penalty_XY, penalty_YX = calc_penalty(string_X , string_Y)
        # strength = abs(penalty)
        # if penalty>0:
        #     direction = 1
        # if penalty<0:
        #     direction = 0
        # if penalty == 0:
        #     direction = np.random.choice([0,1])
        # strengths.append(strength)
        # directions.append(direction)
        
        # ccc_M_S = CCC.compute(feat_mat_class_1_data[data_instance_no, :], feat_mat_class_0_data[data_instance_no, :], LEN_past=120, ADD_meas=15, STEP_size=60, n_partitions=2)
        # ccc_S_M = CCC.compute(feat_mat_class_0_data[data_instance_no, :], feat_mat_class_1_data[data_instance_no, :], LEN_past=120, ADD_meas=15, STEP_size=60, n_partitions=2)
        Y_X.append(penalty_YX)
        X_Y.append(penalty_XY)
    
    Y_causes_X[ROW] = np.mean(Y_X)
    X_causes_Y[ROW] = np.mean(X_Y)
    
    Y_causes_X_var[ROW] = np.var(Y_X)
    X_causes_Y_var[ROW] = np.var(X_Y)
    
RESULT_PATH_FINAL = PATH + '/' +'CAUSALITY_TESTING-RESULTS' + '/'+ DATA_NAME + '/' 


 

try:
    os.makedirs(RESULT_PATH_FINAL)
except OSError:
    print ("Creation of the result directory %s failed" % RESULT_PATH_FINAL)
else:
    print ("Successfully created the result directory %s" % RESULT_PATH_FINAL)

plt.figure(figsize=(15,10))
plt.plot(COUP_COEFF1, Y_causes_X, '-*k', markersize = 10, label = "Y causes X "+"(" + DATA_NAME +")")
plt.plot(COUP_COEFF1, X_causes_Y, '-or', markersize = 10, label = "X causes Y "+"(" + DATA_NAME +")")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=30)
plt.ylabel('Average LZ-Penality', fontsize=30)
plt.ylim(0, 210)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"average-penalty.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"average-penalty.eps", format='eps', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"average-penalty.png", format='png', dpi=300)
plt.show()


plt.figure(figsize=(15,10))
plt.plot(COUP_COEFF1, Y_causes_X_var, '-*k', markersize = 10, label = "Y causes X "+"(" + DATA_NAME +")")
plt.plot(COUP_COEFF1, X_causes_Y_var, '-or', markersize = 10, label = "X causes Y "+"(" + DATA_NAME +")")
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.grid(True)
plt.xlabel('Coupling Coefficient', fontsize=30)
plt.ylabel('Varaince of LZ-Penality', fontsize=30)
plt.ylim(0, 200)
plt.legend(fontsize = 30)
plt.tight_layout()
plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"variance-penalty.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"variance-penalty.eps", format='eps', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"variance-penalty.png", format='png', dpi=300)
plt.show()

### Error Bar Plots

# Compute standard deviation from variance (optional, depends on the units you want to show)
std_lz_X_to_Y = np.sqrt(X_causes_Y_var)
std_lz_Y_to_X = np.sqrt(Y_causes_X_var)

# Create a figure
plt.figure(figsize=(10, 10))

# Plot the average LZ penalty with error bars for variance (standard deviation)
plt.errorbar(COUP_COEFF1, X_causes_Y, yerr=std_lz_X_to_Y, label="X to Y"+" (" + DATA_NAME +")", fmt='-o', color='blue', capsize=5)
plt.errorbar(COUP_COEFF1, Y_causes_X, yerr=std_lz_Y_to_X, label='Y to X'+" (" + DATA_NAME +")", fmt='-s', color='black', capsize=5)

# Add titles and labels
#plt.title('Average and S of LZ Penalty vs Coupling Coefficient')
plt.xlabel('Coupling Coefficient',fontsize=30)
plt.ylabel('Average LZ Penalty', fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=30)
plt.legend(fontsize = 30)
plt.ylim(0, 200)
# Grid and display
plt.grid(True)
plt.tight_layout()

plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"-error-bar.jpg", format='jpg', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"-error-bar.eps", format='eps', dpi=300)
plt.savefig(RESULT_PATH_FINAL+"LZ-"+DATA_NAME+"-error-bar.png", format='png', dpi=300)

# Show the plot
plt.show()
# Saving Results

np.save(RESULT_PATH_FINAL +'average_penalty_YX.npy', Y_causes_X)
np.save(RESULT_PATH_FINAL +'average_penalty_XY.npy', X_causes_Y)


np.save(RESULT_PATH_FINAL +'variance_penalty_YX.npy', Y_causes_X_var)
np.save(RESULT_PATH_FINAL +'variance_penalty_XY.npy', X_causes_Y_var)

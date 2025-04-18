#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

In this code, AR(5) master slave coupled timeseries data is generated. The coupling
coefficient is varied from 0.1 to 1.0.



"""

## AR(1) Process Data Generation Code
import os
import numpy as np
import matplotlib.pyplot as plt
from AR_code import coupled_AR_lag_generator



a = 0.9
b = 0.9
# x_in = 0.1
# y_in = 0.2
LENGTH = 2500
NOISE_INTENSITY = 0.03
LAG = 100
COUP_COEFF1 = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
#COUP_COEFF = 0.8
TRANSIENT_LENGTH = 500

NUM_INSTANCE_PER_CLASS = 1000

DATA_NAME = 'AR-100'

# Class-0 is the Independent Process
class_0_indep = np.zeros((NUM_INSTANCE_PER_CLASS, LENGTH-TRANSIENT_LENGTH))
# Class-1 is the Dependent Process
class_1_dep = np.zeros((NUM_INSTANCE_PER_CLASS, LENGTH-TRANSIENT_LENGTH))

# Label  = 0 corresponds to independent process = cause
class_0_label = np.zeros((NUM_INSTANCE_PER_CLASS, 1))
# Label = 1 corresponds to dependent process = effect
class_1_label = np.ones((NUM_INSTANCE_PER_CLASS, 1))

# Storing the normalized timeseries generated from AR process.

for COUP_COEFF in COUP_COEFF1:
    for num_instance in range(0, NUM_INSTANCE_PER_CLASS):

        X_dependent, Y_independent = coupled_AR_lag_generator(a, b, COUP_COEFF, LAG, LENGTH, NOISE_INTENSITY, TRANSIENT_LENGTH)
        
        # X_norm_dep = (X_dependent - np.min(X_dependent))/(np.max(X_dependent) - np.min(X_dependent))
        # Y_norm_indep = (Y_independent - np.min(Y_independent))/(np.max(Y_independent) - np.min(Y_independent))
        
        class_0_indep[num_instance, :] = Y_independent
        class_1_dep[num_instance, :] = X_dependent





    PATH = os.getcwd()
    RESULT_PATH = PATH + '/DATA/'  + DATA_NAME + '/' +str(LAG)+'/' + str(COUP_COEFF) +'/'
    
    
    try:
        os.makedirs(RESULT_PATH)
    except OSError:
        print ("Creation of the result directory %s failed" % RESULT_PATH)
    else:
        print ("Successfully created the result directory %s" % RESULT_PATH)
    
    
    import scipy
    from scipy import io
    
    
    io.savemat(RESULT_PATH + 'Y_independent_data_class_0.mat', {'class_0_indep_raw_data': class_0_indep})
    io.savemat(RESULT_PATH + 'Y_independent_label_class_0.mat', {'class_0_indep_raw_data_label': class_0_label})


    io.savemat(RESULT_PATH + 'X_dependent_data_class_1.mat', {'class_1_dep_raw_data': class_1_dep})
    io.savemat(RESULT_PATH + 'X_dependent_label_class_1.mat', {'class_1_dep_raw_data_label': class_1_label})

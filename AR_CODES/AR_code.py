#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Functions for genrating master-slave AR(1) coupled processes
"""

## AR Process classification Code
import numpy as np
import matplotlib.pyplot as plt



def AR_1_data_gen(x_in, y_in, a, b, COUP_COEFF, LENGTH, NOISE_INTENSITY, TRANSIENT_LENGTH):
    '''
    

    Parameters
    ----------
    x_in : TYPE: SCALAR
        DESCRIPTION: initial value of the dependent process.
    y_in : TYPE: SCALAR
        DESCRIPTION: initial value of the independent process.
    a: TYPE: SCALAR
        DESCRIPTION: coefficient of X
    b: TYPE: SCALAR
        DESCRIPTION: coefficient of Y
    COUP_COEFF : TYPE: SCALAR
        DESCRIPTION: Coupling Coefficient 
    LENGTH : TYPE: SCALAR
        DESCRIPTION. length of the timeseries
    NOISE_INTENSITY : TYPE: SCALAR
        DESCRIPTION. noise intensity in the AR Process
    TRANSIENT_LENGTH: TYPE: SCALAR
        DESCRIPTION: Transient length that has to be removed from the AR Process

    Returns
    -------
    X : TYPE: 1D array
        DESCRIPTION: Dependent process of length = LENGTH
    Y : TYPE: 1D array
        DESCRIPTION: Independent process of length = LENGTH
    
    '''
    X = np.zeros(LENGTH)
    Y = np.zeros(LENGTH)
    X[0] = x_in
    Y[0] = y_in
    
    for INDEX in range(1, LENGTH):
        X[INDEX] = a*X[ INDEX - 1 ] + COUP_COEFF*Y[ INDEX - 1 ] + NOISE_INTENSITY * np.random.randn(1)
        Y[INDEX] = b*Y[ INDEX - 1] + NOISE_INTENSITY * np.random.randn(1)
        
    return X[TRANSIENT_LENGTH:], Y[TRANSIENT_LENGTH:]









def Bidir_AR_1_data_gen(x_in, y_in, a, b, COUP_COEFF, LENGTH, NOISE_INTENSITY, TRANSIENT_LENGTH):
    '''
    

    Parameters
    ----------
    x_in : TYPE: SCALAR
        DESCRIPTION: initial value of the dependent process.
    y_in : TYPE: SCALAR
        DESCRIPTION: initial value of the independent process.
    a: TYPE: SCALAR
        DESCRIPTION: coefficient of X
    b: TYPE: SCALAR
        DESCRIPTION: coefficient of Y
    COUP_COEFF : TYPE: SCALAR
        DESCRIPTION: Coupling Coefficient 
    LENGTH : TYPE: SCALAR
        DESCRIPTION. length of the timeseries
    NOISE_INTENSITY : TYPE: SCALAR
        DESCRIPTION. noise intensity in the AR Process
    TRANSIENT_LENGTH: TYPE: SCALAR
        DESCRIPTION: Transient length that has to be removed from the AR Process

    Returns
    -------
    X : TYPE: 1D array
        DESCRIPTION: Dependent process of length = LENGTH
    Y : TYPE: 1D array
        DESCRIPTION: Independent process of length = LENGTH
    
    '''
    X = np.zeros(LENGTH)
    Y = np.zeros(LENGTH)
    X[0] = x_in
    Y[0] = y_in
    
    for INDEX in range(1, LENGTH):
        X[INDEX] = a*X[ INDEX - 1 ] + COUP_COEFF*Y[ INDEX - 1 ] + NOISE_INTENSITY * np.random.randn(1)
        Y[INDEX] = b*Y[ INDEX - 1 ] + COUP_COEFF*X[ INDEX - 1 ] + NOISE_INTENSITY * np.random.randn(1)
        
    return X[TRANSIENT_LENGTH:], Y[TRANSIENT_LENGTH:]
    

def coupled_AR_lag_generator(a, b, COUP_COEFF, LAG, LENGTH, NOISE_INTENSITY, TRANSIENT_LENGTH):
    ## Y causing X
    X = np.random.normal(5,1,LENGTH)
    Y = np.random.normal(5,1,LENGTH)
    #generating x and y
    for t in range(0, LENGTH-LAG):
        error_x = NOISE_INTENSITY * np.random.standard_normal()
        error_y = NOISE_INTENSITY * np.random.standard_normal()
        Y[t+LAG] = b * Y[t+LAG-1] + error_y
        X[t+LAG] = a * X[t+LAG-1] + COUP_COEFF * Y[t]  + error_x
        # Y[t+LAG] = b * Y[t] + error_y
        # X[t+LAG] = a * X[t] + COUP_COEFF * Y[t]  + error_x
    return X[TRANSIENT_LENGTH:], Y[TRANSIENT_LENGTH:]


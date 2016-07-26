# pvcMetrics.py
# cates@sci.utah.edu

# This module defines PVC Workgroup Validation Metrics. For a full
# description of these metrics, including the data format, please
# see the file pvcMetric.pdf that should accompany this code.

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


# This metric measures the relative mean squared error (RMSE) in space
# averaged across time.  That is, the RMSE obtained per each time
# instance averaged. Inputs are two potentials matrices (EGM), # as
# defined in pvcMetrics.pdf
def RMSE(X,Y):
    # First define matrix sizes.
    M = X.shape[0] # number of nodes
    T = X.shape[1] # number of time points (samples)

    # Check that matrices are equal dimensions
    if (Y.shape[0] != M) or (Y.shape[1] != T):
        return -1

    # Compute relative error at each time point
    mse = np.zeros(T)
    
    for t in range(0,T):
        norm_y = 0.0;

        for m in range(0,M):
            diff = X[m,t] - Y[m,t]
            mse[t] += diff * diff 
            norm_y += Y[m,t] * Y[m,t]

        mse[t] = mse[t] / norm_y
        

    # Average normalized covariance across all time points
    avgRMSE  = np.mean(mse,0)
    sdevRMSE = np.std(mse,0)

    return (avgRMSE, sdevRMSE, mse)


# corrEGM measures the average correlation in space between true and
# measured potentials.  Inputs are two potentials matrices (EGM), as
# defined in pvcMetrics.pdf
def corrEGM(X,Y):
    # First define matrix sizes.
    M = X.shape[0] # number of nodes
    T = X.shape[1] # number of time points (samples)

    # Check that matrices are equal dimensions
    if (Y.shape[0] != M) or (Y.shape[1] != T):
        return -1

    # Average value at each time point
    x_mean = np.mean(X, axis = 0)
    y_mean = np.mean(Y, axis = 0)

    # Compute normalized corrleation at each time point
    cov = np.zeros(T)
    
    for t in range(0,T):
        norm_x = 0.0;
        norm_y = 0.0;

        for m in range(0,M):
            x_residual = X[m,t] - x_mean[t]
            y_residual = Y[m,t] - y_mean[t]
            cov[t] += x_residual * y_residual 
            norm_x += x_residual * x_residual
            norm_y += y_residual * y_residual

        cov[t] = cov[t] / (np.sqrt(norm_x) * np.sqrt(norm_y))
        

    # Average normalized correlation across all time points
    corrEGM = np.mean(cov,0)
        
    return (corrEGM, x_mean, y_mean, cov)


#This metric measures the correlation between reconstructed and true
#activation times. It is defined similarly to the correlation in EGM,
def corrACTT(X,Y):
    # First define matrix sizes.
    M = X.shape[0] # number of nodes
 
    # Check that matrices are equal dimensions
    if (Y.shape[0] != M):
        return -1

    # Average value at each time point
    x_mean = np.mean(X, axis = 0)
    y_mean = np.mean(Y, axis = 0)

    # Compute normalized covariance
    corr = 0
    
    norm_x = 0.0
    norm_y = 0.0
    
    for m in range(0,M):
        x_residual = X[m] - x_mean
        y_residual = Y[m] - y_mean
        corr += x_residual * y_residual 
        norm_x += x_residual * x_residual
        norm_y += y_residual * y_residual
        
    corr = corr / (np.sqrt(norm_x) * np.sqrt(norm_y))
        
    return (corr)

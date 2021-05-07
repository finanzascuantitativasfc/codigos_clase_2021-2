# -*- coding: utf-8 -*-
"""
Created on Fri May  7 09:13:01 2021

@author: Meva
"""

# import libraries and functions
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

nb_decimals = 6 # 3 4 5 6
scale = 252 # 1 252
notional = 15 # mnUSD

rics = ['BARC.L','BBVA.MC','BNP.PA','CBK.DE','CSGN.SW','DBK.DE',\
        'GLE.PA','HSBA.L','SAN.MC','UBSG.SW','XLF']
    
# rics = ['BP.L','ENI.MI','RDSa.AS','RDSa.L','EQNR.OL','REP.MC','XOP',\
#         'SGRE.MC','VWS.CO','ORSTED.CO','FSLR','NEE']


# # compute covariance matrix via np.cov
# returns = []
# for ric in rics:
#     t = file_functions.load_timeseries(ric)
#     x = t['return_close'].values
#     returns.append(x)
# mtx_covar = np.cov(returns) # cov = covariance 
# mtx_correl = np.corrcoef(returns) # corrcoef = correlation
# # did not work: we need to synchronise timeseries first


# compute variance-covariance matrix by pairwise covariances
size = len(rics)
mtx_covar = np.zeros([size,size])
mtx_correl = np.zeros([size,size])
vec_returns = np.zeros([size,1])
vec_volatilities = np.zeros([size,1])
returns = []
for i in range(size):
    ric_x = rics[i]
    for j in range(i+1):
        ric_y = rics[j]
        t = file_functions.load_synchronised_timeseries(ric_x, ric_y)
        ret_x = t['return_x'].values
        ret_y = t['return_y'].values
        returns = [ret_x, ret_y]
        # covariances
        temp_mtx = np.cov(returns)
        temp_covar = scale*temp_mtx[0][1]
        temp_covar = np.round(temp_covar,nb_decimals)
        mtx_covar[i][j] = temp_covar
        mtx_covar[j][i] = temp_covar
        # correlations
        temp_mtx = np.corrcoef(returns)
        temp_correl = temp_mtx[0][1]
        temp_correl = np.round(temp_correl,nb_decimals)
        mtx_correl[i][j] = temp_correl
        mtx_correl[j][i] = temp_correl
        if j == 0:
            temp_ret = ret_x
    # returns
    temp_mean = np.round(scale*np.mean(temp_ret), nb_decimals)
    vec_returns[i] = temp_mean
    # volatilities
    temp_volatility = np.round(np.sqrt(scale)*np.std(temp_ret), nb_decimals)
    vec_volatilities[i] = temp_volatility
    

# compute eigenvalues and eigenvectors for symmetric matrices
eigenvalues, eigenvectors = LA.eigh(mtx_covar)


print('----')
print('Securities:')
print(rics)
print('----')
print('Returns (annualised):')
print(vec_returns)
print('----')
print('Volatilities (annualised):')
print(vec_volatilities)
print('----')
print('Variance-covariance matrix (annualised):')
print(mtx_covar)
print('----')
print('Correlation matrix:')
print(mtx_correl)

print('----')
print('Eigenvalues:')
print(eigenvalues)
print('----')
print('Eigenvectors:')
print(eigenvectors)


# min-variance portfolio
print('----')
print('Min-variance portfolio:')
print('notional (mnUSD) = ' + str(notional))
variance_explained = eigenvalues[0] / sum(abs(eigenvalues))
eigenvector = eigenvectors[:,0]
port_min_var = notional * eigenvector / sum(abs(eigenvector))
delta_min_var = sum(port_min_var)
print('delta (mnUSD) = ' + str(delta_min_var))
print('variance explained = ' + str(variance_explained))


# PCA (max-variance) portfolio
print('----')
print('PCA portfolio (max-variance):')
print('notional (mnUSD) = ' + str(notional))
variance_explained = eigenvalues[-1] / sum(abs(eigenvalues))
eigenvector = eigenvectors[:,-1]
port_pca = notional * eigenvector / sum(abs(eigenvector))
delta_pca = sum(port_pca)
print('delta (mnUSD) = ' + str(delta_pca))
print('Variance explained = ' + str(variance_explained))
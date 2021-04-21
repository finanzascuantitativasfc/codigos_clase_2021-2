# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:39:52 2021

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

# input parameters
benchmark = '^STOXX50E'
security = 'BBVA.MC' # Reuters Identification Code
hedge_securities =  ['GLE.PA','BNP.PA']
delta_portfolio = 10 # mn USD

# compute betas
capm = file_classes.capm_manager(benchmark, security)
capm.load_timeseries()
capm.compute()
beta_portfolio = capm.beta
beta_portfolio_usd = beta_portfolio * delta_portfolio # mn USD

# print input
print('------')
print('Input portfolio:')
print('Delta mnUSD for ' + security + ' is ' + str(delta_portfolio))
print('Beta for ' + security + ' vs ' + benchmark + ' is ' + str(beta_portfolio))
print('Beta mnUSD for ' + security + ' vs ' + benchmark + ' is ' + str(beta_portfolio_usd))

# compute betas for the hedges
shape = [len(hedge_securities)]
betas = np.zeros(shape)
counter = 0
print('------')
print('Input hedges:')
for hedge_security in hedge_securities:
    capm = file_classes.capm_manager(benchmark, hedge_security)
    capm.load_timeseries()
    capm.compute()
    beta = capm.beta
    print('Beta for hedge[' + str(counter) + '] = ' + hedge_security + ' vs ' + benchmark + ' is ' + str(beta))
    betas[counter] = beta
    counter += 1
    
# exact solution using matrix algebra
deltas = np.ones(shape)
targets = -np.array([[delta_portfolio],[beta_portfolio_usd]])
mtx = np.transpose(np.column_stack((deltas,betas)))
optimal_hedge = np.linalg.inv(mtx).dot(targets)
hedge_delta = np.sum(optimal_hedge)
hedge_beta_usd = np.transpose(betas).dot(optimal_hedge).item()

# print result
print('------')
print('Optimisation result')
print('------')
print('Delta portfolio: ' + str(delta_portfolio))
print('Beta portfolio USD: ' + str(beta_portfolio_usd))
print('------')
print('Delta hedge: ' + str(hedge_delta))
print('Beta hedge USD: ' + str(hedge_beta_usd))
print('------')
print('Optimal hedge:')
print(optimal_hedge)
print('------')



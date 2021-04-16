# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:07:45 2021

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


# capm = file_classes.capm_manager(inputs)
# capm.load_timeseries()
# capm.compute()
# capm.plot()
# print(capm)

# inputs
benchmark = 'BBVA.MC' # variable x
security = 'SAN.MC' # variable y

# load synchronised timeseries
t = file_functions.load_synchronised_timeseries(ric_x=benchmark, ric_y=security)

# linear regression
x = t['return_x'].values
y = t['return_y'].values
slope, intercept, r_value, p_value, std_err = linregress(x,y)

nb_decimals = 8
slope = np.round(slope, nb_decimals)
intercept = np.round(intercept, nb_decimals)
p_value = np.round(p_value, nb_decimals) 
null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
r_value = np.round(r_value, nb_decimals) # correlation coefficient
r_squared = np.round(r_value**2, nb_decimals) # pct of variance of y explained by x
predictor_linreg = intercept + slope*x

# scatterplot of returns
str_title = 'Scatterplot of returns' + '\n'\
    + 'Linear regression | security ' + security\
    + ' | benchmark ' + benchmark + '\n'\
    + 'alpha (intercept) ' + str(intercept)\
    + ' | beta (slope) ' + str(slope) + '\n'\
    + 'p-value ' + str(p_value)\
    + ' | null hypothesis ' + str(null_hypothesis) + '\n'\
    + 'r-value (correl) ' + str(r_value)\
    + ' | r-squared ' + str(r_squared)
plt.figure()
plt.title(str_title)
plt.scatter(x,y)
plt.plot(x, predictor_linreg, color='green')
plt.ylabel(security)
plt.xlabel(benchmark)
plt.grid()
plt.show()


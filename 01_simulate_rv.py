# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:22:02 2021

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

print('Hello World!')

nb_sims = 10**6
df = 500
dist_name = 'normal' # student normal exponential uniform chi-square
dist_type = 'simulated RV' # real custom


if dist_name == 'normal':
    x = np.random.standard_normal(nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'exponential':
    x = np.random.standard_exponential(nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'uniform':
    x = np.random.uniform(0,1,nb_sims)
    x_description = dist_type + ' ' + dist_name
elif dist_name == 'student':
    x = np.random.standard_t(df=df, size=nb_sims)
    x_description = dist_type + ' ' + dist_name + ' | df = ' + str(df)
elif dist_name == 'chi-square':
    x = np.random.chisquare(df=df, size=nb_sims)
    x_description = dist_type + ' ' + dist_name + ' | df = ' + str(df)
    
'''
Goal: create a Jarque-Bera normality test
'''

x_mean = np.mean(x)
x_std = np.std(x)
x_skew = skew(x)
x_kurtosis = kurtosis(x) # excess kurtosis
x_jb_stat = nb_sims/6*(x_skew**2 + 1/4*x_kurtosis**2)
x_p_value = 1 - chi2.cdf(x_jb_stat, df=2)
x_is_normal = (x_p_value > 0.05) # equivalently jb < 6

print('skewness is ' + str(x_skew))
print('kurtosis is ' + str(x_kurtosis))
print('JB statistic is ' + str(x_jb_stat))
print('p-value ' + str(x_p_value))
print('is normal ' + str(x_is_normal))

# plot histogram
plt.figure()
plt.hist(x,bins=100)
plt.title(x_description)
plt.show()


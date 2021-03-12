# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 09:33:38 2021

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2


# get market data
# remember to modify the path to match your own directory
directory = 'C:\\Users\Meva\\.spyder-py3\\data\\2021-2\\'
# inputs
ric = 'VWS.CO' # BBVA.MC MXN=X ^STOXX50E
path = directory + ric + '.csv' 
raw_data = pd.read_csv(path)

# create table of returns
t = pd.DataFrame()
t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
t['close'] = raw_data['Close']
t.sort_values(by='date', ascending=True)
t['close_previous'] = t['close'].shift(1)
t['return_close'] = t['close']/t['close_previous'] - 1
t = t.dropna()
t = t.reset_index(drop=True)

# plot timeseries of price
plt.figure()
plt.plot(t['date'],t['close'])
plt.title('Time series real prices ' + ric)
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()

    
'''
Goal: create a Jarque-Bera normality test
'''
x = t['return_close'].values
x_description = 'market data ' + ric
nb_rows = len(x)

x_mean = np.mean(x)
x_std = np.std(x)
x_skew = skew(x)
x_kurtosis = kurtosis(x) # excess kurtosis
x_jb_stat = nb_rows/6*(x_skew**2 + 1/4*x_kurtosis**2)
x_p_value = 1 - chi2.cdf(x_jb_stat, df=2)
x_is_normal = (x_p_value > 0.05) # equivalently jb < 6

# jb_list = []
# jb_list.append(x_jb_stat)

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
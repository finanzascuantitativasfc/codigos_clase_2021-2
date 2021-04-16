# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:11:49 2021

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)


def load_timeseries(ric):
    
    directory = 'C:\\Users\Meva\\.spyder-py3\\data\\2021-2\\' # hardcoded
    path = directory + ric + '.csv' 
    raw_data = pd.read_csv(path)
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(raw_data['Date'], dayfirst=True)
    t['close'] = raw_data['Close']
    t.sort_values(by='date', ascending=True)
    t['close_previous'] = t['close'].shift(1)
    t['return_close'] = t['close']/t['close_previous'] - 1
    t = t.dropna()
    t = t.reset_index(drop=True)
    
    return t


def load_synchronised_timeseries(ric_x, ric_y):
    
    # get timeseries of x and y
    table_x = file_functions.load_timeseries(ric_x)
    table_y = file_functions.load_timeseries(ric_y)
    # synchronize timestamps
    timestamps_x = list(table_x['date'].values)
    timestamps_y = list(table_y['date'].values)
    timestamps = list(set(timestamps_x) & set(timestamps_y))
    # synchronised time series for x
    table_x_sync = table_x[table_x['date'].isin(timestamps)]
    table_x_sync.sort_values(by='date', ascending=True)
    table_x_sync = table_x_sync.reset_index(drop=True)
    # synchronised time series for y
    table_y_sync = table_y[table_y['date'].isin(timestamps)]
    table_y_sync.sort_values(by='date', ascending=True)
    table_y_sync = table_y_sync.reset_index(drop=True)
    # table of returns for x and y
    t = pd.DataFrame()
    t['date'] = table_x_sync['date']
    t['price_x'] = table_x_sync['close']
    t['return_x'] = table_x_sync['return_close']
    t['price_y'] = table_y_sync['close']
    t['return_y'] = table_y_sync['return_close']
    
    return t
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 09:15:32 2021

@author: Meva
"""
import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2


class distribution_manager():
    
    
    def __init__(self, inputs):
        self.inputs = inputs
        self.data_table = None
        self.description = None
        self.nb_rows = None
        
        
    def load_timeseries(self):
        
        data_type = self.inputs['data_type']
        
        if data_type == 'simulation':
            
            nb_sims = self.inputs['nb_sims']
            dist_name = self.inputs['variable_name']
            degrees_freedom = self.inputs['degrees_freedom']
            
            if dist_name == 'normal':
                x = np.random.standard_normal(nb_sims)
                x_description = data_type + ' ' + dist_name
            elif dist_name == 'exponential':
                x = np.random.standard_exponential(nb_sims)
                x_description = data_type + ' ' + dist_name
            elif dist_name == 'uniform':
                x = np.random.uniform(0,1,nb_sims)
                x_description = data_type + ' ' + dist_name
            elif dist_name == 'student':
                x = np.random.standard_t(df=degrees_freedom, size=nb_sims)
                x_description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
            elif dist_name == 'chi-square':
                x = np.random.chisquare(df=degrees_freedom, size=nb_sims)
                x_description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
       
            self.description = x_description
            self.nb_rows = nb_sims
            self.vec_returns = x
       
        elif data_type == 'real':
            
            directory = 'C:\\Users\Meva\\.spyder-py3\\data\\2021-2\\'
            ric = self.inputs['variable_name']
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
            
            self.data_table = t
            self.description = 'market data ' + ric
            self.nb_rows = t.shape[0]
            self.vec_returns = t['return_close'].values
            
            
    def plot_histogram(self):
        
        plt.figure()
        plt.hist(self.vec_returns,bins=100)
        plt.title(self.description)
        plt.show()

        
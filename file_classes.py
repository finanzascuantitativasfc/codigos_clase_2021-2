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

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)


class distribution_manager():
    
    
    def __init__(self, inputs):
        self.inputs = inputs
        self.data_table = None
        self.description = None
        self.nb_rows = None
        self.vec_returns = None
        self.mean = None
        self.std = None
        self.skew = None
        self.kurtosis = None # excess kurtosis
        self.jb_stat = None # under normality of self.vec_returns this distributes as chi-square with 2 degrees of freedom
        self.p_value = None # equivalently jb < 6
        self.is_normal = None
        self.sharpe = None
        self.var_95 = None
        self.percentile_25 = None
        self.median = None
        self.percentile_75 = None
        
        
    def __str__(self):
        str_self = self.description + ' | size ' + str(self.nb_rows) + '\n' + self.plot_str()
        return str_self
        
        
    def load_timeseries(self):
        
        # data_type = self.inputs['data_type']
        data_type = self.inputs.data_type
        
        if data_type == 'simulation':
            
            nb_sims = self.inputs.nb_sims
            dist_name = self.inputs.variable_name
            degrees_freedom = self.inputs.degrees_freedom
            
            if dist_name == 'normal':
                x = np.random.standard_normal(nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'exponential':
                x = np.random.standard_exponential(nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'uniform':
                x = np.random.uniform(0,1,nb_sims)
                self.description = data_type + ' ' + dist_name
            elif dist_name == 'student':
                x = np.random.standard_t(df=degrees_freedom, size=nb_sims)
                self.description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
            elif dist_name == 'chi-square':
                x = np.random.chisquare(df=degrees_freedom, size=nb_sims)
                self.description = data_type + ' ' + dist_name + ' | df = ' + str(degrees_freedom)
       
            self.nb_rows = nb_sims
            self.vec_returns = x
       
        elif data_type == 'real':
            
            ric = self.inputs.variable_name
            t = file_functions.load_timeseries(ric)
            
            self.data_table = t
            self.description = 'market data ' + ric
            self.nb_rows = t.shape[0]
            self.vec_returns = t['return_close'].values
            
            
    def plot_histogram(self):
        plt.figure()
        plt.hist(self.vec_returns,bins=100)
        plt.title(self.description)
        plt.xlabel(self.plot_str())
        plt.show()
        
        
    def compute(self):
        self.mean = np.mean(self.vec_returns)
        self.std = np.std(self.vec_returns)
        self.skew = skew(self.vec_returns)
        self.kurtosis = kurtosis(self.vec_returns) # excess kurtosis
        self.jb_stat = self.nb_rows/6*(self.skew**2 + 1/4*self.kurtosis**2)
        self.p_value = 1 - chi2.cdf(self.jb_stat, df=2)
        self.is_normal = (self.p_value > 0.05) # equivalently jb < 6
        self.sharpe = self.mean / self.std * np.sqrt(252) # annualised
        self.var_95 = np.percentile(self.vec_returns,5)
        self.cvar_95 = np.mean(self.vec_returns[self.vec_returns <= self.var_95])
        self.percentile_25 = self.percentile(25)
        self.median = np.median(self.vec_returns)
        self.percentile_75 = self.percentile(75)

        
    def plot_str(self):
        nb_decimals = 4
        plot_str = 'mean ' + str(np.round(self.mean,nb_decimals))\
            + ' | std dev ' + str(np.round(self.std,nb_decimals))\
            + ' | skewness ' + str(np.round(self.skew,nb_decimals))\
            + ' | kurtosis ' + str(np.round(self.kurtosis,nb_decimals)) + '\n'\
            + 'Jarque Bera ' + str(np.round(self.jb_stat,nb_decimals))\
            + ' | p-value ' + str(np.round(self.p_value,nb_decimals))\
            + ' | is normal ' + str(self.is_normal) + '\n'\
            + 'Sharpe annual ' + str(np.round(self.sharpe,nb_decimals))\
            + ' | VaR 95% ' + str(np.round(self.var_95,nb_decimals))\
            + ' | CVaR 95% ' + str(np.round(self.cvar_95,nb_decimals)) + '\n'\
            + 'percentile 25% ' + str(np.round(self.percentile_25,nb_decimals))\
            + ' | median ' + str(np.round(self.median,nb_decimals))\
            + ' | percentile 75% ' + str(np.round(self.percentile_75,nb_decimals))
        return plot_str
    
    
    def percentile(self, pct):
        percentile = np.percentile(self.vec_returns,pct)
        return percentile
    
    
class distribution_input():
    
    def __init__(self):
        self.data_type = None # simulation real custom
        self.variable_name = None # normal student exponential chi-square uniform VWS.CO
        self.degrees_freedom = None # only used in simulation + student and chi-square
        self.nb_sims = None # only in simulation

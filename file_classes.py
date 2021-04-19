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
from scipy.stats import skew, kurtosis, chi2, linregress

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
    
    
class capm_manager():
    
    def __init__(self, benchmark, security):
        self.benchmark = benchmark # variable x
        self.security = security # variable y
        self.nb_decimals = 4
        self.data_table = None
        self.slope = None
        self.intercept = None
        self.p_value = None
        self.null_hypothesis = None
        self.r_value = None
        self.std_err = None
        self.predictor_linreg = None
        
        
    def __str__(self):
        return self.str_self()
    
    
    def str_self(self):
        str_self = 'Linear regression | security ' + self.security\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.intercept)\
            + ' | beta (slope) ' + str(self.slope) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'r-value (correl) ' + str(self.r_value)\
            + ' | r-squared ' + str(self.r_squared)
        return str_self
    
    
    def load_timeseries(self):
        self.data_table = file_functions.load_synchronised_timeseries(ric_x=self.benchmark, ric_y=self.security)
    
    
    def compute(self):
        # linear regression
        x = self.data_table['return_x'].values
        y = self.data_table['return_y'].values
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        self.slope = np.round(slope, self.nb_decimals)
        self.intercept = np.round(intercept, self.nb_decimals)
        self.p_value = np.round(p_value, self.nb_decimals) 
        self.null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
        self.r_value = np.round(r_value, self.nb_decimals) # correlation coefficient
        self.r_squared = np.round(r_value**2, self.nb_decimals) # pct of variance of y explained by x
        self.predictor_linreg = intercept + slope*x
        
        
    def plot_timeseries(self):
        plt.figure(figsize=(12,5))
        plt.title('Time series of prices')
        plt.xlabel('Time')
        plt.ylabel('Prices')
        ax = plt.gca()
        ax1 = self.data_table.plot(kind='line', x='date', y='price_x', ax=ax, grid=True,\
                                  color='blue', label=self.benchmark)
        ax2 = self.data_table.plot(kind='line', x='date', y='price_y', ax=ax, grid=True,\
                                  color='red', secondary_y=True, label=self.security)
        ax1.legend(loc=2)
        ax2.legend(loc=1)
        plt.show()
    
    
    def plot_linear_regression(self):
        x = self.data_table['return_x'].values
        y = self.data_table['return_y'].values
        str_title = 'Scatterplot of returns' + '\n' + self.str_self()
        plt.figure()
        plt.title(str_title)
        plt.scatter(x,y)
        plt.plot(x, self.predictor_linreg, color='green')
        plt.ylabel(self.security)
        plt.xlabel(self.benchmark)
        plt.grid()
        plt.show()
        
    
class distribution_input():
    
    def __init__(self):
        self.data_type = None # simulation real custom
        self.variable_name = None # normal student exponential chi-square uniform VWS.CO
        self.degrees_freedom = None # only used in simulation + student and chi-square
        self.nb_sims = None # only in simulation
        
    

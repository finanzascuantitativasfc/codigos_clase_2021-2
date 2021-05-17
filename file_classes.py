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
from scipy.optimize import minimize
from numpy import linalg as LA

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)
from scipy.optimize import minimize


class distribution_manager(): 
    
    def __init__(self, inputs):
        self.inputs = inputs # distribution_inputs
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
        
    
    
class capm_manager():
    
    def __init__(self, benchmark, security):
        self.benchmark = benchmark # variable x
        self.security = security # variable y
        self.nb_decimals = 4
        self.data_table = None
        self.alpha = None
        self.beta = None
        self.p_value = None
        self.null_hypothesis = None
        self.correlation = None
        self.r_squared = None
        self.std_err = None
        self.predictor_linreg = None
        
        
    def __str__(self):
        return self.str_self()
    
    
    def str_self(self):
        str_self = 'Linear regression | security ' + self.security\
            + ' | benchmark ' + self.benchmark + '\n'\
            + 'alpha (intercept) ' + str(self.alpha)\
            + ' | beta (slope) ' + str(self.beta) + '\n'\
            + 'p-value ' + str(self.p_value)\
            + ' | null hypothesis ' + str(self.null_hypothesis) + '\n'\
            + 'correl (r-value) ' + str(self.correlation)\
            + ' | r-squared ' + str(self.r_squared)
        return str_self
    
    
    def load_timeseries(self):
        self.data_table = file_functions.load_synchronised_timeseries(ric_x=self.benchmark, ric_y=self.security)
    
    
    def compute(self):
        # linear regression
        x = self.data_table['return_x'].values
        y = self.data_table['return_y'].values
        slope, intercept, r_value, p_value, std_err = linregress(x,y)
        self.alpha = np.round(intercept, self.nb_decimals)
        self.beta = np.round(slope, self.nb_decimals)
        self.p_value = np.round(p_value, self.nb_decimals) 
        self.null_hypothesis = p_value > 0.05 # p_value < 0.05 --> reject null hypothesis
        self.correlation = np.round(r_value, self.nb_decimals) # correlation coefficient
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



class hedge_manager():
    
    def __init__(self, inputs):
        self.inputs = inputs # hedge_inputs
        self.benchmark = inputs.benchmark # the market in CAPM, in general ^STOXX50E
        self.security = inputs.security # portfolio to hedge
        self.hedge_securities = inputs.hedge_securities # hedge universe
        self.nb_hedges = len(self.hedge_securities)
        self.portfolio_delta = inputs.delta_portfolio
        self.portfolio_beta = None
        self.portfolio_beta_usd = None
        self.betas = None
        self.optimal_hedge = None
        self.hedge_delta = None
        self.hedge_beta_usd = None
        self.regularisation = 0.0
        
    
    def load_betas(self):
        benchmark = self.benchmark
        security = self.security
        hedge_securities = self.hedge_securities
        portfolio_delta = self.portfolio_delta
        # compute beta for the portfolio
        capm = file_classes.capm_manager(benchmark, security)
        capm.load_timeseries()
        capm.compute()
        portfolio_beta = capm.beta
        portfolio_beta_usd = portfolio_beta * portfolio_delta # mn USD
        # print input
        print('------')
        print('Input portfolio:')
        print('Delta mnUSD for ' + security + ' is ' + str(portfolio_delta))
        print('Beta for ' + security + ' vs ' + benchmark + ' is ' + str(portfolio_beta))
        print('Beta mnUSD for ' + security + ' vs ' + benchmark + ' is ' + str(portfolio_beta_usd))
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
        
        self.portfolio_beta = portfolio_beta
        self.portfolio_beta_usd = portfolio_beta_usd
        self.betas = betas
        
        
    def compute(self, regularisation=0.0):
        # numerical solution
        dimensions = len(self.hedge_securities)
        x = np.zeros([dimensions,1])
        betas = self.betas
        optimal_result = minimize(fun=file_functions.cost_function_hedge, x0=x, args=(self.portfolio_delta, self.portfolio_beta_usd, betas, regularisation))
        self.optimal_hedge = optimal_result.x
        self.hedge_delta = np.sum(self.optimal_hedge)
        self.hedge_beta_usd = np.transpose(betas).dot(self.optimal_hedge).item()
        self.regularisation = regularisation
        self.print_result('numerical')
        
        
    def compute_exact(self):
        # exact solution using matrix algebra
        dimensions = len(self.hedge_securities)
        if dimensions != 2:
            print('------')
            print('Cannot compute exact solution because dimensions = ' + str(dimensions) + ' =/= 2')
            return
        deltas = np.ones([dimensions])
        betas = self.betas
        targets = -np.array([[self.delta_portfolio],[self.beta_portfolio_usd]])
        mtx = np.transpose(np.column_stack((deltas,betas)))
        self.optimal_hedge = np.linalg.inv(mtx).dot(targets)
        self.hedge_delta = np.sum(self.optimal_hedge)
        self.hedge_beta_usd = np.transpose(betas).dot(self.optimal_hedge).item()
        self.print_result('exact')
        
    
    def print_result(self, algo_type):
        print('------')
        print('Optimisation result - ' + algo_type + ' solution')
        print('------')
        print('Delta portfolio: ' + str(self.portfolio_delta))
        print('Beta portfolio USD: ' + str(self.portfolio_beta_usd))
        print('------')
        print('Delta hedge: ' + str(self.hedge_delta))
        print('Beta hedge USD: ' + str(self.hedge_beta_usd))
        print('------')
        print('Optimal hedge:')
        print(self.optimal_hedge)
        

        
class hedge_input:
   
   def __init__(self):
       self.benchmark = None # the market in CAPM, in general ^STOXX50E
       self.security = 'BBVA.MC' # portfolio to hedge
       self.hedge_securities =  ['^STOXX50E','^FCHI'] # hedge universe
       self.delta_portfolio = None # in mn USD, default 10   



class portfolio_manager:
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.size = len(rics)
        self.notional = notional
        self.nb_decimals = 6
        self.scale = 252
        self.covariance_matrix = None
        self.correlation_matrix = None
        self.returns = None
        self.volatilities = None
        
        
    def compute_covariance_matrix(self, bool_print=True):
        # compute variance-covariance matrix by pairwise covariances
        rics = self.rics
        size = self.size
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
                temp_covar = self.scale*temp_mtx[0][1]
                temp_covar = np.round(temp_covar,self.nb_decimals)
                mtx_covar[i][j] = temp_covar
                mtx_covar[j][i] = temp_covar
                # correlations
                temp_mtx = np.corrcoef(returns)
                temp_correl = temp_mtx[0][1]
                temp_correl = np.round(temp_correl,self.nb_decimals)
                mtx_correl[i][j] = temp_correl
                mtx_correl[j][i] = temp_correl
                if j == 0:
                    temp_ret = ret_x
            # returns
            temp_mean = np.round(self.scale*np.mean(temp_ret), self.nb_decimals)
            vec_returns[i] = temp_mean
            # volatilities
            temp_volatility = np.round(np.sqrt(self.scale)*np.std(temp_ret), self.nb_decimals)
            vec_volatilities[i] = temp_volatility
        # compute eigenvalues and eigenvectors for symmetric matrices
        eigenvalues, eigenvectors = LA.eigh(mtx_covar)
        
        self.covariance_matrix = mtx_covar
        self.correlation_matrix = mtx_correl
        self.returns = vec_returns
        self.volatilities = vec_volatilities
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        
        if bool_print:
            print('----')
            print('Securities:')
            print(self.rics)
            print('----')
            print('Returns (annualised):')
            print(self.returns)
            print('----')
            print('Volatilities (annualised):')
            print(self.volatilities)
            print('----')
            print('Variance-covariance matrix (annualised):')
            print(self.covariance_matrix)
            print('----')
            print('Correlation matrix:')
            print(self.correlation_matrix)
            print('----')
            print('Eigenvalues:')
            print(self.eigenvalues)
            print('----')
            print('Eigenvectors:')
            print(self.eigenvectors)
            
            
    def compute_portfolio(self, portfolio_type='default', target_return=None):
        
        portfolio = portfolio_item(self.rics, self.notional)
        
        if portfolio_type == 'min-variance':
            portfolio.type = portfolio_type
            portfolio.variance_explained = self.eigenvalues[0] / sum(abs(self.eigenvalues))
            eigenvector = self.eigenvectors[:,0]
            if max(eigenvector) < 0:
                eigenvector = - eigenvector
            weights_normalised = eigenvector / sum(abs(eigenvector))
            
        elif portfolio_type == 'pca':
            portfolio.type = portfolio_type
            portfolio.variance_explained = self.eigenvalues[-1] / sum(abs(self.eigenvalues))
            eigenvector = self.eigenvectors[:,-1]
            if max(eigenvector) < 0:
                eigenvector = - eigenvector
            weights_normalised = eigenvector / sum(abs(eigenvector))
            
        elif portfolio_type == 'default' or portfolio_type == 'equi-weight':
            portfolio.type = 'equi-weight'
            weights_normalised = 1 / self.size * np.ones([self.size])
            
        elif portfolio_type == 'markowitz':
            portfolio.type = portfolio_type
            if target_return == None:
                target_return = np.mean(self.returns)
            portfolio.target_return = target_return    
            # initialise optimisation
            x = np.zeros([self.size,1])
            # initialise constraints
            cons = [{"type": "eq", "fun": lambda x: np.transpose(self.returns).dot(x).item() - target_return},\
                    {"type": "eq", "fun": lambda x: sum(abs(x)) - 1}]
            bnds = [(0, None) for i in range(self.size)]
            # compute optimisation
            res = minimize(file_functions.compute_portfolio_variance, x, args=(self.covariance_matrix), constraints=cons, bounds=bnds)
            weights_normalised = res.x
        
        weights = self.notional * weights_normalised
        portfolio.weights = weights
        portfolio.delta = sum(weights)
        portfolio.pnl_annual = np.transpose(self.returns).dot(weights).item()
        portfolio.return_annual = np.transpose(self.returns).dot(weights_normalised).item()
        portfolio.volatility_annual = file_functions.compute_portfolio_volatility(weights_normalised, self.covariance_matrix)
        portfolio.sharpe_annual = portfolio.return_annual / portfolio.volatility_annual
        
        return portfolio
 
            

class portfolio_item():
    
    def __init__(self, rics, notional):
        self.rics = rics
        self.notional = notional
        self.type = ''
        self.weights = []
        self.delta = 0.0
        self.variance_explained = None
        self.pnl_annual = None
        self.target_return = None
        self.return_annual = None
        self.volatility_annual = None
        self.sharpe_annual = None


    def summary(self):
        print('-----')
        print('Portfolio type: ' + self.type)
        print('Rics:')
        print(self.rics)
        print('Weights:')
        print(self.weights)
        print('Notional (mnUSD): ' + str(self.notional))
        print('Delta (mnUSD): ' + str(self.delta))
        if not self.variance_explained == None:
            print('Variance explained: ' + str(self.variance_explained))
        if not self.pnl_annual == None:
            print('Profit and loss annual: ' + str(self.pnl_annual))
        if not self.target_return == None:
            print('Target return: ' + str(self.target_return))
        if not self.return_annual == None:
            print('Return annual: ' + str(self.return_annual))
        if not self.volatility_annual == None:
            print('Volatility annual: ' + str(self.volatility_annual))
        if not self.sharpe_annual == None:
            print('Sharpe ratio annual: ' + str(self.sharpe_annual))
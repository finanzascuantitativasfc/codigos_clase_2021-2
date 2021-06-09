# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:55:01 2021

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


# universe
rics = ['BARC.L','BBVA.MC','BNP.PA','CBK.DE','CSGN.SW','DBK.DE',\
        'GLE.PA','HSBA.L','SAN.MC','UBSG.SW']
# rics = ['BP.L','ENI.MI','RDSa.AS','RDSa.L','EQNR.OL','REP.MC','XOP']
# rics = ['SGRE.MC','VWS.CO','ORSTED.CO','FSLR','NEE']
# rics = ['^GSPC','^VIX']
# rics = ['^FTSE','^GDAXI','^FCHI','^STOXX50E']
# rics = ['CAD=X','CHF=X','CNY=X','EURUSD=X','GBPUSD=X',\
#         'JPY=X','MXN=X','NOK=X','SEK=X']

# input params
notional = 300 # mnUSD
target_return = 0.015 # 0.01 0.04 0.36 0.6 0.05 0.015
include_min_variance=False

# efficient frontier
dict_portfolios = file_functions.compute_efficient_frontier(rics, notional, target_return, include_min_variance)


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
from scipy.optimize import minimize

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

# inputs
inputs = file_classes.hedge_input()
inputs.benchmark = '^STOXX50E'
# inputs.security = 'ENI.MI'
inputs.security = 'BBVA.MC'
# inputs.hedge_securities =  ['BP.L','ENI.MI','RDSa.AS','RDSa.L','EQNR.OL','REP.MC','XOP']
# inputs.hedge_securities =  ['EQNR.OL','REP.MC']
inputs.hedge_securities =  ['^GDAXI','^FCHI']
# inputs.hedge_securities =  ['^STOXX50E','^GDAXI','^FCHI']
# inputs.hedge_securities =  ['^STOXX50E']
inputs.delta_portfolio = 10 # mn USD

# computations
hedge = file_classes.hedge_manager(inputs)
hedge.load_betas() # get the betas for portfolio and hedges
hedge.compute(regularisation=0.01) # numerical solution
hedge_delta = hedge.hedge_delta
hedge_beta_usd = hedge.hedge_beta_usd
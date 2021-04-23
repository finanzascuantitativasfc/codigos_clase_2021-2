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

# import our own files and reload
import file_classes
importlib.reload(file_classes)
import file_functions
importlib.reload(file_functions)

# inputs
inputs = file_classes.hedge_input()
inputs.benchmark = '^STOXX50E'
inputs.security = 'RDSa.AS' # Reuters Identification Code
inputs.hedge_securities =  ['FP.PA','BP.L']
inputs.delta_portfolio = 10 # mn USD

# computations
hedge = file_classes.hedge_manager(inputs)
hedge.load_betas() # get the betas for portfolio and hedges
hedge.compute() # compute optimal hedge via CAPM

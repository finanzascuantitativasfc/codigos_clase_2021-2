# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 09:06:25 2021

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
import file_functions
importlib.reload(file_functions)
import file_classes
importlib.reload(file_classes)

# inputs
inputs = file_classes.option_input()
inputs.price = 102
inputs.time = 0.0 # in years
inputs.volatility = 0.25
inputs.interest_rate = 0.01
inputs.maturity = 2/12 # in years
inputs.strike = 100
inputs.call_or_put = 'put'

price_black_scholes_put = file_functions.compute_price_black_scholes(inputs)


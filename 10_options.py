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
inputs.price = 19.7862
inputs.time = 0.0 # in years
inputs.volatility = 0.1442
inputs.interest_rate = 0.0158
inputs.maturity = 3/12 # in years
inputs.strike = 19.7862
inputs.call_or_put = 'call'
number_simulations = 1*10**6

# price using Black-Scholes formula
price_black_scholes = file_functions.compute_price_black_scholes(inputs)

# price using Monte Carlo simulations
prices_monte_carlo = file_functions.compute_price_monte_carlo(inputs, number_simulations)
print(prices_monte_carlo)
prices_monte_carlo.plot_histogram()





# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 09:42:34 2021

@author: Meva
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

data_type = 'real' # simulation real custom
variable_name = 'VWS.CO' # normal student VWS.CO

dm = class_distribution_manager()

dm.load_timeseries(data_type, variable_name, bool_plot=True) # polymorphism
dm.compute()
dm.plot_histogram()

print(dm)
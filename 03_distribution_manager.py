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

# import our own files and reload
import file_classes
importlib.reload(file_classes)

inputs = {
"data_type" : 'simulation', # simulation real custom
"variable_name" : 'normal', # normal student exponential chi-square
"degrees_freedom" : 2, # only used in student and chi-square
"nb_sims" : 10**6
}

dm = file_classes.distribution_manager(inputs)
dm.load_timeseries() # polymorphism
dm.plot_histogram()

# dm.compute()
# print(dm)
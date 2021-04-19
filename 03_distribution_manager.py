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

inputs = file_classes.distribution_input()
inputs.data_type = 'real'
inputs.variable_name = '^STOXX50E'
inputs.degrees_freedom = 9
inputs.nb_sims = 10**6

dm = file_classes.distribution_manager(inputs) # initialise constructor
dm.load_timeseries() # get the timeseries
dm.compute() # compute returns and all different risk metrics
dm.plot_histogram() # plot histogram
print(dm) # write all data in console



# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 09:07:45 2021

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
benchmark = '^GDAXI' # variable x
security = '^FCHI' # variable y

capm = file_classes.capm_manager(benchmark, security)
capm.load_timeseries()
# capm.plot_timeseries()
capm.compute()
capm.plot_linear_regression()
print(capm)


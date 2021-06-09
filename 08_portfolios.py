# -*- coding: utf-8 -*-
"""
Created on Wed May 12 09:12:38 2021

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

notional = 300 # mnUSD
rics = ['BARC.L','BBVA.MC','BNP.PA','CBK.DE','CSGN.SW','DBK.DE',\
        'GLE.PA','HSBA.L','SAN.MC','UBSG.SW']
# rics = ['BARC.L','BBVA.MC','BNP.PA','CBK.DE','CSGN.SW','DBK.DE',\
#         'GLE.PA','HSBA.L','SAN.MC','UBSG.SW','XLF']
# rics = ['BARC.L','BNP.PA','XLF']
# rics = ['^GSPC','^VIX']
# rics = ['BP.L','ENI.MI','RDSa.AS','RDSa.L','EQNR.OL','REP.MC','XOP',\
#         'SGRE.MC','VWS.CO','ORSTED.CO','FSLR','NEE']
# rics = ['BP.L','ENI.MI','RDSa.AS','RDSa.L','EQNR.OL','REP.MC','XOP']
# rics = ['SGRE.MC','VWS.CO','ORSTED.CO','FSLR','NEE']

port_mgr = file_classes.portfolio_manager(rics, notional)
port_mgr.compute_covariance_matrix(bool_print=True)

port_min_var = port_mgr.compute_portfolio('min-variance')
port_min_var.summary()

port_min_var_l1 = port_mgr.compute_portfolio('min-variance-l1')
port_min_var_l1.summary()

port_min_var_l2 = port_mgr.compute_portfolio('min-variance-l2')
port_min_var_l2.summary()

port_long_only = port_mgr.compute_portfolio('long-only')
port_long_only.summary()

port_pca = port_mgr.compute_portfolio('pca')
port_pca.summary()

port_equi = port_mgr.compute_portfolio('equi-weight')
port_equi.summary()

port_volatility = port_mgr.compute_portfolio('volatility-weighted')
port_volatility.summary()

port_markowitz = port_mgr.compute_portfolio('markowitz', target_return=None)
port_markowitz.summary()




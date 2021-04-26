# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 09:43:49 2021

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

# # define the function to minimise
# def cost_function(x):
#     f = (x[0] - 7.0)**2 + (x[1] + 5)**2 + (x[2] - 13)**2
#     return f

# # initialise optimisation
# x = np.zeros([3,1])

# # compute optimisation
# optimal_result = scipy.optimize.minimize(fun=cost_function, x0=x)

# # print
# print('------')
# print('Optimisation result:')
# print(optimal_result)

# define the function to minimise
def cost_function(x, roots, coeffs):
    f = 0
    for n in range(len(x)):
        f += coeffs[n]*(x[n] - roots[n])**2
    return f

# input parameters
dimensions = 5
roots = np.random.randint(low=-20, high=20, size=dimensions)
coeffs = np.ones([dimensions,1])

# initialise optimisation
x = np.zeros([dimensions,1])

# compute optimisation
optimal_result = minimize(fun=cost_function, x0=x, args=(roots,coeffs))

# print
print('------')
print('Optimisation result:')
print(optimal_result)
print('------')
print('Roots:')
print(roots)
print('------')

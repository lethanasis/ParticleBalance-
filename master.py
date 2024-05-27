# -*- coding: utf-8 -*-
"""
Created on Wed May  8 13:55:09 2024

@author: thana
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve
import sys
import time
from scipy.constants import c, e
from IonHandler import IonHandler
import ionrate
from DistributionFunction import DistributionFunction
import h5py
from Bisection import power_balance, initial_guess, newton_method
from rewriting_matrices import construct_F, construct_Jacobian, Zeff

"""
Add system variables:
:param n_Ne:    Neon density
:param pn:      neutral pressure
"""

n_Ne = 1e19
pn = 0.1 
NRE=8E15
sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM\py\DREAM\Formulas')

#Full parth to HDF5 file
filename = 'cache\data_ionization'

with h5py.File(filename, "r") as f:
    INTD = f['H']['data'][:]
    INTNe = f['Ne']['data'][:]
    
ions = IonHandler()
ions.addIon('D', 1 , 1e19)
ions.addIon('Ne', 10, n_Ne)
ions['D'].IonThreshold = INTD
ions['Ne'].IonThreshold = INTNe


'''Set up distribution function'''
fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=400)
ions.cacheImpactIonizationRate(fre)

Te = 2 
Tn = Te*e
nfree, n = ionrate.equilibriumAtPressure(ions, pn, Te, Tn, fre)
ions.setSolution(n)
#print(f'master {n}')
a = 1 
b = 10
Di = 1
dist = 0.25*1.2



Te = initial_guess(power_balance, a, b, Di=Di, dist=dist, ions=ions)
print(Te)
ne = 5e19
n, ne, Te, i = newton_method(ions, ne, Te)

#print(f'number of iter {i+1}')
# =============================================================================
# 
# print(f'Densities are {n}')
# print(f'Electron density is {ne}')
# print(f'Electron temperature is {Te}')
# print(f'Number of iterations: {i}')
# =============================================================================

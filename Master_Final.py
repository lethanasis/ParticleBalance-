# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:05:36 2024

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
from Rewriting_nonlinear import newton_method, bisection, power_balance
#from rewriting_matrices import construct_F, construct_Jacobian, Zeff
from Matrices_final import Zeff, construct_F, construct_Jacobian


"""
Add system variables:
:param n_Ne:    Neon density
:param pn:      neutral pressure
"""
Volume = 4.632

n_Ne = 7.2e18/Volume
#n_Ne = 1e19
pn   = 5
#pn = np.logspace(np.log10(2e-2), 0.5, 20)
NRE  = 1.6e16
#NRE = 8e15

sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM\py\DREAM\Formulas')

'''Set up ion species'''
#Full parth to HDF5 file
filename = 'cache\data_ionization'

with h5py.File(filename, "r") as f:
    INTD = f['H']['data'][:]
    INTNe = f['Ne']['data'][:]
    
ions = IonHandler()
ions.addIon('D', 1 , 1e20)
ions.addIon('Ne', 10, n_Ne)
ions['D'].IonThreshold = INTD
ions['Ne'].IonThreshold = INTNe


'''Set up distribution function'''
fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=400)
ions.cacheImpactIonizationRate(fre)


Te = 2
Tn = Te*e
ne, n = ionrate.equilibriumAtPressure(ions, pn, Te, Tn, fre)
ions.setSolution(n)
a = 1
b = 3.6
Di =1 
dist = 0.25*1.2

Te = bisection(power_balance, a, b, ions, pn, Te, Tn, fre, Di, dist, NRE)
print(Te)

Tn = Te*e
ne , n = ionrate.equilibriumAtPressure(ions, pn, Te, Tn, fre)
ions.setSolution(n)
#ne=5e19

Z = Zeff(ions, ne)
n, ne, Te, i = newton_method(ions, pn, ne, Te, fre, Z, Di, dist, NRE)

print(f'Temperature is : {Te}')
print(f'Densities are : {n}')
print(f'Electron density is : {ne}')
print(f'Iterations {i+1}')

# =============================================================================
# plt.figure()
# plt.plot(range(len(n_arr)), n_arr, label = 'Power Balance')
# plt.legend()
# plt.grid(True)
# 
# plt.figure()
# plt.plot(range(len(ne_arr)), ne_arr, label = 'dx - ne')
# plt.legend()
# plt.grid(True)
# 
# plt.figure()
# plt.plot(range(len(Te_arr)), Te_arr, label = 'dx - n_D-1')
# plt.legend()
# plt.grid(True)
# =============================================================================

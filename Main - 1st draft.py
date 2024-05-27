# -*- coding: utf-8 -*-
"""
Created on Thu May 16 15:16:02 2024

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
pn = np.logspace(np.log10(2e-2), 0.5, 20)
NRE  = 1.6e16
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

Di = 1
dist = 0.25*1.2
Te_arr = []
for i in range(pn.size):
    Te = 0.7284 + 0.351/np.sqrt(pn[i])
    Tn = Te*e
    ne, n = ionrate.equilibriumAtPressure(ions, pn[i], Te, Tn, fre)
    ions.setSolution(n)
    a = 0.5
    b = 50
    while True:
        try:
            Te = bisection(power_balance, a, b, ions, pn[i], Te, Tn, fre, Di, dist, NRE)
            print(Te)
            break
        except Exception as l:
            b-=1
    Tn = Te*e
    ne , n = ionrate.equilibriumAtPressure(ions, pn[i], Te, Tn, fre)
    ions.setSolution(n)
    

    Z = Zeff(ions, ne)
    n, ne, Te, i = newton_method(ions, pn[i], ne, Te, fre, Z, Di, dist, NRE)
    Te_arr.append(Te)
Te_exp = []
for i in range(pn.size):
    Te_exp.append(0.3418/np.sqrt(pn[i]) + 0.7478)
    
    
    
plt.plot(pn, Te_arr, label = 'Te', marker = 'o' )
plt.plot(pn, Te_exp, label = 'Exper.', marker = '*')
plt.legend()
plt.grid(True)
plt.xscale('log')
plt.xlabel('Neutral Pressure [Pa]')
plt.ylabel('Te [eV]')
    


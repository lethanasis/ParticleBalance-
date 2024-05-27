# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:41:46 2024

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
from get_derivatives import evaluateBraamsConductivity, getEc, braams_conductivity_derivative_with_respect_to_Z, getCoulombLogarithm, derivative_coulomb_log_T, derivative_sigma_T, getCriticalFieldDerivativeWithRespectToTemperature, derivative_sigma_T
from Radiation_losses import RadiationLosses, Transport
import matplotlib.pyplot as plt


Volume = 4.632

#n_Ne = 7.2e18/Volume

n_Ne = np.linspace(0.1*7.2e18, 7.2e18, 5)
pn = np.logspace(np.log10(2e-2), 0.5, 5)
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

# Other imports and code remain the same...

# Diagnostic function for power_balance_test
def power_balance_test(Te, ne, ions: IonHandler, Di, dist, NRE):
    Z = Zeff(ions, ne)
    Ec = getEc(Te, ne)
    sigma = evaluateBraamsConductivity(ne, Te, Z)
    Prad, Prad_prime = RadiationLosses(ions, ne, Te)
    Ptransp, Ptransp_prime = Transport(ions, Di, dist, Te, Tw=0.025)
    
    re_heat = e * c * NRE * Ec
    ohm_heat = sigma * Ec**2
    rad_loss = Prad
    neut_transp = Ptransp
    
    F = re_heat + ohm_heat - rad_loss - neut_transp
    
    # Ensure all return values are scalars
    return float(F), float(re_heat), float(ohm_heat), float(rad_loss), float(neut_transp)

Te_arr = np.linspace(1, 10, 100)

for i in reversed(range(pn.size)):
    for j in reversed(range(n_Ne.size)):
        F_arr = []
        re_heat_arr = []
        ohm_heat_arr = []
        rad_loss_arr = []
        neut_transp_arr = []
        exception_occurred = False
        
        for Te in Te_arr:
            try:
                Tn = Te * e
                ions['Ne'].n = n_Ne[j]
                ne, n = ionrate.equilibriumAtPressure(ions, pn[i], Te, Tn, fre)
                ions.setSolution(n)
                
                # Check and print shapes to debug
                print(f"Shapes - Te: {Te}, ne: {ne.shape if hasattr(ne, 'shape') else type(ne)}, n: {n.shape if hasattr(n, 'shape') else type(n)}")
                
                F, re_heat, ohm_heat, rad_loss, neut_transp = power_balance_test(Te, ne, ions, Di, dist, NRE)
                
                # Ensure values are scalar before appending
                F_arr.append(F)
                re_heat_arr.append(re_heat)
                ohm_heat_arr.append(ohm_heat)
                rad_loss_arr.append(rad_loss)
                neut_transp_arr.append(neut_transp)
            
            except Exception as k:
                exception_occurred = True
                F_arr.append(np.nan)
                re_heat_arr.append(np.nan)
                ohm_heat_arr.append(np.nan)
                rad_loss_arr.append(np.nan)
                neut_transp_arr.append(np.nan)
                continue  # Move to the next iteration if an exception occurs
        
        print(f' i = {i}, j = {j}')
        plt.figure()
        plt.plot(Te_arr, F_arr, label=r'Power Balance')
        plt.plot(Te_arr, re_heat_arr, label=r'RE heating')
        plt.plot(Te_arr, ohm_heat_arr, label=r'Ohmic Heating')
        plt.plot(Te_arr, rad_loss_arr, label=r'Radiation losses')
        plt.plot(Te_arr, neut_transp_arr, label=r'Neutral Transport')
        plt.xlabel(r'$T_e$ [e]')
        plt.ylabel(r'Power [W]')
        plt.title(rf'{{\texttt{{Power balance \& terms over temperature for pn = {pn[i]:.2f}, $n_{{\texttt{{Ne}}}}$ = {n_Ne[j]}}}}}')
        plt.grid(True)
        plt.legend()
        plt.yscale('symlog')
        plt.yticks([-1e9, -1e6, -1e3, 0, 1e3, 1e6, 1e9])
        plt.show()

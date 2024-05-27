# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 10:17:23 2024

@author: thana
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys
import scipy.constants
import scipy.interpolate
import scipy.special
from scipy.constants import c,e, m_e, epsilon_0, pi
from Radiation_losses import RadiationLosses, Transport
from IonHandler import IonHandler
from DistributionFunction import DistributionFunction
import ionrate


sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM\py\DREAM\Formulas')

from  PlasmaParameters import evaluateBraamsConductivity,getConnorHastieCriticalField
from get_derivatives import derivative_coulomb_log_T, derivative_sigma_T, getCriticalFieldDerivativeWithRespectToTemperature
from ITER import NRE


# Full path to the HDF5 file
filename = 'cache\data_ionization'

# Get ionization threshold for all ions from the HDF5 file. 
with h5py.File(filename, "r") as f:
    INTD = f['H']['data'][:]
    INTNe = f['Ne']['data'][:]

ne = 1e19


initial_guess = 2000
Te=initial_guess
max_iter = 1000
tol = 1e-3

'''Initialize ions'''
ions = IonHandler()
ions.addIon('D', 1, 1e19)
ions.addIon('Ne', 10, 1e19)
ions['D'].IonThreshold = INTD
ions['Ne'].IonThreshold  = INTNe

'''Set up distribution function'''
fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=400)
ions.cacheImpactIonizationRate(fre)





pn = 0.1
nfree, n = ionrate.equilibriumAtPressure(ions, pn, 2, 2*scipy.constants.e, fre)
Di = 1
Dist = 1.2 * 0.25
Tw = 0.025

ions.setSolution(n)

print(f'n={n}, nfree = {nfree}')
for ion in ions:
    for j in range(ion.Z+1):
        print(f'Density of {ion.name} is {ion.solution[j]}')

def PowerBalance(Te, nfree):
    """Power Balance equation to solve"""
    Ec = getConnorHastieCriticalField(Te, nfree)
    #print(f'Conner Hastie Critical field is {Ec}')
    cond = evaluateBraamsConductivity(nfree, Te, 1)
    sigma = cond[0]
    dlogL = derivative_coulomb_log_T(Te, nfree)
    dcond = derivative_sigma_T(nfree, Te, 1)
    dsigma = dcond[0]
    dEc, dEc2 = getCriticalFieldDerivativeWithRespectToTemperature(Te, nfree)
    #print(f'Derivative and derivative square of Ec is {dEc} and {dEc2}')
    first = ( (e**4*nfree*NRE)/(4* pi *epsilon_0 * c * m_e) ) * dlogL
    second  = dsigma * Ec**2 + sigma * dEc2
    #print(f'First term in the derivative is {first}')
    #print(f'Second term in the derivative is {second}')
    ### TEST
    if 2==0:
        print(f"Type dsig = {type(dsigma)}")
        print(f"Type EC = {type(Ec)}")
        print(f"Type sigma = {type(sigma)}")
        print(f"Type dEc2 = {type(dEc2)}")
        
    Prad, Prad_prime = RadiationLosses(ions, ne, Te, nfree)
    Prad = Prad[0]
    Prad_prime = Prad_prime[0]
    Prad_prime = Prad_prime[0]
    #print(f'Prad_prime = {Prad_prime}')
    Ptransp, Ptransp_prime = Transport(ions, Di, Dist, Te, Tw)
    #print(f'Ptrasp_prime = {Ptransp_prime}')
    
    if 2==0:   
        print(f'Runaway term is :{e*c*NRE*Ec}')
        print(f'Ohmic term is : {sigma[0]*(Ec**2)}')
        print(f'Radiation term is : {Prad}')
        print(f'Transport term is : {Ptransp}')
    
    return e*c*NRE*Ec +sigma*(Ec**2)-Prad - Ptransp, first+second -Prad_prime - Ptransp_prime , Ec, sigma


def Newton_Method(initial_guess,max_iter,tol):
    """Solve equilibrium using Newton-Raphson method"""
    Te=initial_guess
    for i in range(max_iter):     
        f, fprime = PowerBalance(Te, nfree)
        print(f'power balance is {f}, derivative is {fprime}')
        #print(fprime)
        delta_Te = -f/fprime
        print(delta_Te)
        Te = Te+delta_Te
        print(f'Te = {Te}')
        #print(Te)
        if abs(delta_Te) < tol:
            print(f"Root found at Te = {Te} after {i+1} iterations")
            return Te
        else: print("Maximum iterations reached, no root found")
        return None
    


Te = np.linspace(1, 5, 2000)
f = np.zeros(Te.shape)
for i in range(Te.size):
    f[i] = PowerBalance(Te[i], nfree)[0]
    
plt.plot(Te, f)
plt.show()
#root = Newton_Method(initial_guess, max_iter, tol)


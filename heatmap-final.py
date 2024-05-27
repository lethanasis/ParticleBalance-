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
n_Ne = np.linspace(0.1*7.2e18, 7.2e18, 20)
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
ions.addIon('Ne', 10, 7.2e18)
ions['D'].IonThreshold = INTD
ions['Ne'].IonThreshold = INTNe


'''Set up distribution function'''
fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=400)
ions.cacheImpactIonizationRate(fre)

Di = 1
dist = 0.25*1.2


#Te_arr = np.linspace(1, 10, 100)

# =============================================================================
# for Te in Te_arr:
#     Tn = Te*e
#     ne, n = ionrate.equilibriumAtPressure(ions, pn, Te, Tn, fre)
# 
# Te_arr = []
# =============================================================================
Te_mtx = np.zeros(((pn.size, n_Ne.size)))

for j in range(n_Ne.size):
    for i in range(pn.size):
        Te = 0.7284 + 0.351/np.sqrt(pn[i])
        Tn = 0.8*Te*e
        ions['Ne'].n = n_Ne[j]
        ne, n = ionrate.equilibriumAtPressure(ions, pn[i], Te, Tn, fre)
        ions.setSolution(n)
        a=0.5
        b=5
        print(f'i={i}, j ={j}')
        #print(f'j={j}')
        while True:
            try:
                ne, n = ionrate.equilibriumAtPressure(ions, pn[i], a, a*e, fre)
                break
            except Exception as l:
                print(f'Adjusting value a due to exception : {l}')
                #b-=1
                a+=0.1
        while True:
            try:
                ne, n = ionrate.equilibriumAtPressure(ions, pn[i], b, b*e, fre)
                #print(b)
                break
            except Exception as l:
                print(f'Adjusting value b due to exception : {l}')
                b-=1
        if a>b:
            continue
        Te = bisection(power_balance, a, b, ions, pn[i], Te, Tn, fre, Di, dist, NRE)       
        Tn = 0.8*Te*e
        ne , n = ionrate.equilibriumAtPressure(ions, pn[i], Te, Tn, fre)
        ions.setSolution(n)
        
        Z = Zeff(ions, ne)
        n, ne, Te, k = newton_method(ions, pn[i], ne, Te, fre, Z, Di, dist, NRE)
        if Te<0:
            continue
        Te_mtx[j,i] = Te

# =============================================================================
# plt.imshow(Te_mtx, cmap = 'hot', interpolation='nearest',extent=[pn.min(), pn.max(), n_Ne.min(), n_Ne.max()], aspect='auto')
# cbar=plt.colorbar()
# cbar.set_label('Te [eV]')
# plt.xlabel('Neutral Pressure [Pa]')
# plt.ylabel('Neon Density ')
# plt.show()
# =============================================================================

# Use pcolormesh to create the heatmap
# Note: pcolormesh requires the dimensions of the data to match the coordinates
# Plot the heatmap
X, Y = np.meshgrid(pn, n_Ne)
heatmap = plt.pcolormesh(X, Y, Te_mtx, cmap='hot')

# Set labels and title
plt.colorbar(heatmap, label='Te [eV]')
plt.xlabel('Neutral Pressure [Pa]')
plt.ylabel('Neon Density')

# Set the x-axis to logarithmic scale
plt.xscale('log')

# Show the plot
plt.show()


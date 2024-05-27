# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 14:33:34 2024

@author: thana
"""

import sys

sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\DREAM')
sys.path.append(r'C:\Users\thana\OneDrive\Desktop\Masters Shit\Thesis\particlebalance')
import h5py
from IonHandler import IonHandler
from DistributionFunction import DistributionFunction
import numpy as np
import config
import ionrate
import matplotlib.pyplot as plt

from scipy.constants import c, e



ions = IonHandler()
"""
ions.addIon('D', 1, 2e19)
#ions.addIon('H', 1, 2e19)
ions.addIon('Ne', 10, 7.2e18/4.632)
#ions.addIon('Ar', 18, 1e15)
"""


VOLUME = 840
nD0 = 1e20

injD  = 2e24 / VOLUME
#injNe = 1e23 / VOLUME
injNe = 1e22 / VOLUME

ions.addIon('D', Z=1, n=nD0+injD)
ions.addIon('Ne', Z=10, n=injNe)

NRE = 5e6 / (e*c*2**2*np.pi)
pn = np.logspace(np.log10(2e-2), 0.5, 20)

fre = DistributionFunction()
fre.setStep(nre=NRE, pMin=1, pMax=100, pUpper=40, nP=40)

def temperature(pn):
    """
    Evaluate electron temperature as a function of neutral pressure.
    (fitted to TCV experimental data)
    """
    return 0.7284 + 0.351/np.sqrt(pn)
    #return 0.7284 + 0.351/np.sqrt(pn) + 0.5
    #return 2*0.7284 + 0.351/np.sqrt(pn)


ions.cacheImpactIonizationRate(fre)

"""Assumened neutral pressure"""
Tn = e*.8

 
nfree = []
nfree0 = []
n = []
n0 = []
Te = []
I = []
R = []
Ire = []
 


for i in range(pn.size):
    _Te = temperature(pn[i])
    _nfree, _n = ionrate.equilibriumAtPressure(ions=ions, pn=pn[i], Te=_Te, Tn=Tn, fre=fre)
    #print(f'{i+1}... ', end="" if (i+1)%10!=0 else '\n', flush=True)
    print(f"_nfree={_nfree}")
    I.append(ions['D'].scd(Z0=0, n=_nfree, T=_Te)*_nfree)
    R.append(ions['D'].acd(Z0=1, n=_nfree, T=_Te)*_nfree)
    Ire.append(ions['D'].evaluateImpactIonizationRate(Z0=0, fre=fre))
    
    # Without fre
    _n0 = ionrate.equilibrium(ions, Te=_Te, fre=None)
    _nfree0 = ions.getElectronDensity()

    Te.append(_Te)
    nfree.append(_nfree)
    n.append(_n)
    nfree0.append(_nfree0)
    n0.append(_n0)
    
Te = np.array(Te)
nfree = np.array(nfree)
n = np.array(n)
nfree0 = np.array(nfree0)
n0 = np.array(n0)
I = np.array(I)
R = np.array(R)
Ire = np.array(Ire)

with h5py.File('output/ITER.h5', 'w') as f:
        f['nfree'] = nfree
        f['nfree0'] = nfree0
        f['ni'] = n
        f['ni0'] = n0
        f['pn'] = pn
        f['Z'] = np.array([ion.Z for ion in ions])
        f['Te'] = Te
        f['DI'] = I
        f['DR'] = R
        f['DIre'] = Ire

fig, ax = plt.subplots(1, 1, figsize=(5, 3))

ax.loglog(pn, nfree, 'k', label=r'w/ $f_{\sf re}$')
ax.loglog(pn, nfree0, 'r--', label=r'w/o $f_{\sf re}$')
ax.set_xlabel('Neutral pressure (Pa)')
ax.set_ylabel('Electron density (m$^{-3}$)')
ax.set_xlim([min(pn), max(pn)])
ax.legend(frameon=False)

fig.tight_layout()
plt.show()

"""
if __name__ == '__main__':
    sys.exit(main())
"""







#nn=ions.getNeutralDensity()
#ne=ions.getElectronDensity()

